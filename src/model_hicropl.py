import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision

from src.hicropl import (
    TextEncoder,
    VisualEncoder,
    VisualVisualPromptLearner,
    SimpleTextPromptLearner,
)


def freeze_model(m):
    """Freeze all parameters of the given module."""
    for param in m.parameters():
        param.requires_grad_(False)


def freeze_all_but_bn(model):
    """Freeze every parameter except those owned by nn.LayerNorm modules.

    Matches by module membership (not attribute name) so it correctly covers
    parameters that aren't literally named `weight`/`bias`, e.g.
    nn.MultiheadAttention's `in_proj_weight`/`in_proj_bias`, or loose
    nn.Parameters like `class_embedding`/`positional_embedding`/`proj`/
    `text_projection`/`logit_scale` — all of which must stay frozen per the
    CLIP-AT design (only LayerNorm trainable; Attention and MLP frozen).
    """
    ln_param_ids = {
        id(p)
        for m in model.modules() if isinstance(m, torch.nn.LayerNorm)
        for p in m.parameters()
    }
    for p in model.parameters():
        if id(p) not in ln_param_ids:
            p.requires_grad_(False)


def unfreeze_ln(m):
    """Mở lại weight/bias của mọi LayerNorm trong module.

    Dùng SAU `freeze_model(...)` để thực thi pattern "chỉ LN trainable":
        freeze_model(encoder)           # đông cứng tất cả
        encoder.apply(unfreeze_ln)      # chỉ mở LN
    """
    if isinstance(m, nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(True)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(True)


class CustomCLIP(nn.Module):
    """
    HiCroPL-SBIR Architecture Wrapper.
    Sử dụng HiCroPLFeatureExtractor làm nòng cốt.
    """

    def __init__(self, cfg, clip_model, classnames=None):
        super().__init__()
        self.cfg = cfg

        if classnames is None:
            classnames = []
        if len(classnames) == 0:
            raise ValueError("CustomCLIP requires non-empty classnames during initialization.")

        original_device = next(clip_model.parameters()).device
        self.dtype = clip_model.dtype

        # 1. Single shared backbone for both photo and sketch (matches ducta/baseline:
        # one CLIP copy, same LayerNorm weights updated by gradients from both modalities).
        self.clip = copy.deepcopy(clip_model).to(original_device)
        freeze_all_but_bn(self.clip)

        # Print trainable param counts for verification
        total = sum(p.numel() for p in self.clip.parameters())
        trainable = sum(p.numel() for p in self.clip.parameters() if p.requires_grad)
        print(f"clip (shared): trainable {trainable:,} / total {total:,} params")

        # Single shared logit scale (matches ducta/baseline)
        self.logit_scale = self.clip.logit_scale

        # -- Prompt Learners --
        # Initialize Visual-Visual learner + simple text learners + adapters
        print("Initializing Visual Prompt Learner (photo + sketch, independent)...")
        self.visual_visual_learner = VisualVisualPromptLearner(cfg, self.clip, self.clip)

        print("Initializing Photo Text Prompt Learner...")
        cfg_photo = copy.copy(cfg)
        cfg_photo.ctx_init = getattr(cfg, 'ctx_init', 'a photo of a')
        self.text_prompt_photo = SimpleTextPromptLearner(cfg_photo, classnames, self.clip)

        print("Initializing Sketch Text Prompt Learner...")
        cfg_sketch = copy.copy(cfg)
        cfg_sketch.ctx_init = getattr(cfg, 'ctx_init_sketch', 'a sketch of a')
        self.text_prompt_sketch = SimpleTextPromptLearner(cfg_sketch, classnames, self.clip)

        # -- Encoders (both branches wrap the SAME shared backbone) --
        self.text_encoder_photo = TextEncoder(self.clip)
        self.text_encoder_sketch = TextEncoder(self.clip)
        self.visual_encoder_photo = VisualEncoder(self.clip)
        self.visual_encoder_sketch = VisualEncoder(self.clip)

    def normalize_features(self, feat_prenorm):
        """L2-normalize feature tensors."""
        return feat_prenorm / feat_prenorm.norm(dim=-1, keepdim=True)

    def forward(self, x, classnames):
        """
        Forward pass for training with optimized redundancy.
        Calls visual learner ONCE and routes prompts by branch.
        """
        if len(x) == 5:
            sk_tensor, photo_tensor, neg_tensor, label, _filename = x
        else:
            sk_tensor, photo_tensor, neg_tensor, label = x[:4]

        # 1. Call visual-visual learner ONCE (shared by both branches)
        vis1_shallow, vis2_shallow, vis1_deeper, vis2_deeper = self.visual_visual_learner()

        # 2. Photo branch: text learner + visual routing (vis2)
        # Compute text features for ALL classes (not just batch) - needed for loss computation
        text_input_photo_all, cross_prompts_text_deeper_photo = self.text_prompt_photo(label=None)  # All classes
        text_features_all_photo = self.text_encoder_photo(text_input_photo_all, self.text_prompt_photo.tokenized_prompts, cross_prompts_text_deeper_photo)
        image_features_photo = self.visual_encoder_photo(photo_tensor.type(self.dtype), vis2_shallow, vis2_deeper)

        # 3. Sketch branch: text learner + visual routing (vis1)
        # Compute text features for ALL classes (not just batch) - needed for loss computation
        text_input_sketch_all, cross_prompts_text_deeper_sketch = self.text_prompt_sketch(label=None)  # All classes
        text_features_all_sketch = self.text_encoder_sketch(text_input_sketch_all, self.text_prompt_sketch.tokenized_prompts, cross_prompts_text_deeper_sketch)
        image_features_sketch = self.visual_encoder_sketch(sk_tensor.type(self.dtype), vis1_shallow, vis1_deeper)

        # 4. Negative branch (uses photo encoder + photo visual prompts)
        image_features_neg = self.visual_encoder_photo(neg_tensor.type(self.dtype), vis2_shallow, vis2_deeper)

        # 5. Normalize features
        photo_feat = image_features_photo / image_features_photo.norm(dim=-1, keepdim=True)
        sketch_feat = image_features_sketch / image_features_sketch.norm(dim=-1, keepdim=True)
        neg_feat = image_features_neg / image_features_neg.norm(dim=-1, keepdim=True)
        text_feat_photo = text_features_all_photo / text_features_all_photo.norm(dim=-1, keepdim=True)
        text_feat_sketch = text_features_all_sketch / text_features_all_sketch.norm(dim=-1, keepdim=True)

        # 6. Compute logits
        logit_scale = self.logit_scale.exp()
        logits_photo = logit_scale * photo_feat @ text_feat_photo.t()
        logits_sketch = logit_scale * sketch_feat @ text_feat_sketch.t()

        return (
            photo_feat, logits_photo,
            sketch_feat, logits_sketch,
            neg_feat, label,
            text_feat_photo, text_feat_sketch,
        )


class HiCroPL_SBIR(pl.LightningModule):
    def __init__(self, cfg, args, classnames, model):
        super().__init__()
        self.cfg = cfg
        self.args = args
        self.classnames = classnames
        self.model = model
        
        self.best_metric = 1e-3
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)

        self.test_photo_features = []
        self.test_sketch_features = []
        self.test_photo_labels = []
        self.test_sketch_labels = []

    def on_train_epoch_start(self):
        # NOTE: Encoders stay in training mode (required for LayerNorm to use batch statistics)
        # Setting eval() here would conflict with forward() expectation and break BN/LN behavior
        pass

    def on_fit_start(self):
        """Log the number of learnable prompt tokens per branch once at fit start.

        Logs four scalars (tokens count):
        - `tokens_visual_photo`
        - `tokens_visual_sketch`
        - `tokens_text_photo`
        - `tokens_text_sketch`
        """
        try:
            vv = self.model.visual_visual_learner
            # visual tokens: number of prompt vectors (prompt_depth * n_ctx)
            tokens_visual_photo = len(vv.cross_prompts_photo) * vv.n_ctx
            tokens_visual_sketch = len(vv.cross_prompts_sketch) * vv.n_ctx
        except Exception:
            tokens_visual_photo = 0
            tokens_visual_sketch = 0

        try:
            tp = self.model.text_prompt_photo
            ts = self.model.text_prompt_sketch
            tokens_text_photo = len(tp.cross_prompts_text) * tp.cross_prompts_text[0].shape[0]
            tokens_text_sketch = len(ts.cross_prompts_text) * ts.cross_prompts_text[0].shape[0]
        except Exception:
            tokens_text_photo = 0
            tokens_text_sketch = 0

        # Log to Lightning logger and print for immediate visibility
        self.print(f"Learnable tokens - visual/photo: {tokens_visual_photo}, visual/sketch: {tokens_visual_sketch}, text/photo: {tokens_text_photo}, text/sketch: {tokens_text_sketch}")
        # Use self.log so TensorBoard/other loggers capture these scalars
        # Use rank_zero_only to avoid duplicate logs in distributed runs
        try:
            self.log('tokens_visual_photo', tokens_visual_photo, prog_bar=True, logger=True)
            self.log('tokens_visual_sketch', tokens_visual_sketch, prog_bar=True, logger=True)
            self.log('tokens_text_photo', tokens_text_photo, prog_bar=True, logger=True)
            self.log('tokens_text_sketch', tokens_text_sketch, prog_bar=True, logger=True)
        except Exception:
            # Fallback to print-only if logger not ready
            pass

    def configure_optimizers(self):
        def add_unique_params(candidates, out_list, seen_ids):
            for p in candidates:
                if p.requires_grad and id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    out_list.append(p)

        seen_ids = set()

        prompt_params = []
        # Collect from shared visual learner + per-branch text learners
        # These include all their internal params (CrossPromptAttention, AttentionPooling, etc.)
        add_unique_params(self.model.visual_visual_learner.parameters(), prompt_params, seen_ids)
        add_unique_params(self.model.text_prompt_photo.parameters(), prompt_params, seen_ids)
        add_unique_params(self.model.text_prompt_sketch.parameters(), prompt_params, seen_ids)

        ln_params = []
        # Only collect LayerNorms from clip encoders (NOT from learners, already included above)
        learner_modules = {
            'visual_visual_learner', 'text_prompt_photo', 'text_prompt_sketch'
        }
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                # Skip if inside a learner module (already included with learner params)
                if not any(learner_name in name for learner_name in learner_modules):
                    add_unique_params(module.parameters(recurse=False), ln_params, seen_ids)

        extra_trainable_params = []
        for _, p in self.model.named_parameters():
            if p.requires_grad and id(p) not in seen_ids:
                seen_ids.add(id(p))
                extra_trainable_params.append(p)

        non_prompt_params = ln_params + extra_trainable_params

        self.print(f"Number of trainable prompt params: {sum(p.numel() for p in prompt_params):,}")
        self.print(f"Number of trainable non-prompt params: {sum(p.numel() for p in non_prompt_params):,}")

        prompt_lr = getattr(self.cfg, 'prompt_lr', 1e-5)
        clip_ln_lr = getattr(self.cfg, 'clip_LN_lr', 1e-5)

        param_groups = [{'params': prompt_params, 'lr': prompt_lr}]
        if non_prompt_params:
            param_groups.append({'params': non_prompt_params, 'lr': clip_ln_lr})

        # No weight_decay (matches ducta/baseline's Adam call, which also omits it -> default 0).
        return torch.optim.Adam(param_groups)

    def training_step(self, batch, batch_idx):
        from src.losses_hicropl import loss_fn_hicropl
        features = self.model(batch, self.classnames)
        loss = loss_fn_hicropl(self.args, features)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        
        return loss

    def extract_eval_features(self, tensor, modality):
        """Extract visual features (prompted only, no distill mixing)."""
        # Call visual learner once, cache outputs
        vis1_shallow, vis2_shallow, vis1_deeper, vis2_deeper = self.model.visual_visual_learner()

        if modality == 'photo':
            visual_encoder = self.model.visual_encoder_photo
            vis_shallow, vis_deeper = vis2_shallow, vis2_deeper
        else:
            visual_encoder = self.model.visual_encoder_sketch
            vis_shallow, vis_deeper = vis1_shallow, vis1_deeper

        feat = visual_encoder(tensor.type(self.model.dtype), vis_shallow, vis_deeper)
        return feat / feat.norm(dim=-1, keepdim=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._validation_step_category(batch, batch_idx, dataloader_idx)

    def _validation_step_category(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 3:
            tensor, label, type_data = batch
        else:
            tensor, label = batch
            type_data = None
            
        if dataloader_idx == 0:
            sketch_feat = self.extract_eval_features(tensor, modality='sketch')
            self.test_sketch_features.append(sketch_feat.cpu().detach()) 
            self.test_sketch_labels.append(label.cpu().detach())
        elif dataloader_idx == 1:
            photo_feat = self.extract_eval_features(tensor, modality='photo')
            self.test_photo_features.append(photo_feat.cpu().detach())   
            self.test_photo_labels.append(label.cpu().detach())

    def on_validation_epoch_end(self):
        return self._on_validation_epoch_end_category()

    def _on_validation_epoch_end_category(self):
        if not self.test_photo_features or not self.test_sketch_features:
            self.print("Warning: Missing features for validation. Skipping metrics.")
            return

        gallery_features = torch.cat(self.test_photo_features, dim=0).to(self.device)
        query_features   = torch.cat(self.test_sketch_features, dim=0).to(self.device)
        
        all_photo_category  = torch.cat(self.test_photo_labels, dim=0).to(self.device)
        all_sketch_category = torch.cat(self.test_sketch_labels, dim=0).to(self.device)

        similarity_matrix = query_features @ gallery_features.t()

        dataset = getattr(self.args, 'dataset', 'sketchy')
        if dataset == "sketchy_2" or dataset == "sketchy_ext":
            map_k = 200
            p_k = 200
        elif dataset == "quickdraw":
            map_k = 0
            p_k = 200
        else:
            map_k = 0
            p_k = 100

        ap = torch.zeros(len(query_features), device=self.device)
        precision = torch.zeros(len(query_features), device=self.device)

        for idx in range(len(query_features)):
            category = all_sketch_category[idx]
            distance = similarity_matrix[idx]
            target = (all_photo_category == category)

            if map_k != 0:
                top_k_actual = min(map_k, len(gallery_features))
                ap[idx] = retrieval_average_precision(distance, target, top_k=top_k_actual)
            else:
                ap[idx] = retrieval_average_precision(distance, target)

            precision[idx] = retrieval_precision(distance, target, top_k=p_k)

        mAP = torch.mean(ap)
        mean_precision = torch.mean(precision)

        self.log("mAP", mAP, on_step=False, on_epoch=True)
        self.log(f"P@{p_k}", mean_precision, on_step=False, on_epoch=True)
        self.log("val_mAP", mAP, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"val_P@{p_k}", mean_precision, on_step=False, on_epoch=True)
        self.log("best_mAP", self.best_metric, on_step=False, on_epoch=True, prog_bar=False)

        if map_k != 0:
            self.log(f"val_map_{map_k}", mAP, on_step=False, on_epoch=True)
        else:
            self.log("val_map_all", mAP, on_step=False, on_epoch=True)
        self.log(f"val_p_{p_k}", mean_precision, on_step=False, on_epoch=True)

        if self.global_step > 0:
            self.best_metric = self.best_metric if (self.best_metric > mAP.item()) else mAP.item()

        if map_k != 0:
            self.print('mAP@{}: {:.4f}, P@{}: {:.4f}, Best mAP: {:.4f}'.format(
                map_k, mAP.item(), p_k, mean_precision.item(), self.best_metric))
        else:
            self.print('mAP@all: {:.4f}, P@{}: {:.4f}, Best mAP: {:.4f}'.format(
                mAP.item(), p_k, mean_precision.item(), self.best_metric))

        train_loss = self.trainer.callback_metrics.get("train_loss", None)
        if train_loss is not None:
            self.print(f"Train loss (epoch avg): {train_loss.item():.6f}")

        self.test_photo_features.clear()
        self.test_sketch_features.clear()
        self.test_photo_labels.clear()
        self.test_sketch_labels.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
