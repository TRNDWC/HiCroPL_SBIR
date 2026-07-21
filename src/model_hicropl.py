import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision

from src.hicropl import (
    TextEncoder,
    VisualEncoder,
    VisualPromptLearner,
    VisualVisualPromptLearner,
    TextPromptLearner,
)


def freeze_model(m):
    """Freeze all parameters of the given module."""
    for param in m.parameters():
        param.requires_grad_(False)
        

def freeze_all_but_bn(m):
    """
    Sets requires_grad=False for all parameters except LayerNorm.
    This is usually used with model.apply(freeze_all_but_bn).
    """
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad_(False)


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

        # 1. Branch-specific models (2 deep copies: photo, sketch)
        self.clip_photo = copy.deepcopy(clip_model).to(original_device)
        self.clip_sketch = copy.deepcopy(clip_model).to(original_device)

        self.clip_sketch.apply(freeze_all_but_bn)
        self.clip_photo.apply(freeze_all_but_bn)

        # Optional: fully freeze the CLIP text backbone (transformer,
        # token_embedding, ln_final, positional_embedding, text_projection),
        # overriding freeze_all_but_bn's LN-stays-trainable behavior for text
        # only -- visual branch is untouched. Prompt learner tokens
        # (text_prompt_photo/sketch) are separate modules, unaffected, still
        # trainable -- this only removes the CLIP text backbone's own capacity.
        self.freeze_text = getattr(cfg, 'freeze_text', False)
        if self.freeze_text:
            for clip_branch in (self.clip_photo, self.clip_sketch):
                freeze_model(clip_branch.transformer)
                freeze_model(clip_branch.token_embedding)
                freeze_model(clip_branch.ln_final)
                clip_branch.positional_embedding.requires_grad_(False)
                clip_branch.text_projection.requires_grad_(False)

        # Print trainable param counts per branch for verification
        def _count_trainable(m):
            total = 0
            trainable = 0
            for p in m.parameters():
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()
            return total, trainable

        for name, module in (
            ("clip_photo", self.clip_photo),
            ("clip_sketch", self.clip_sketch),
        ):
            tot, tr = _count_trainable(module)
            print(f"{name}: trainable {tr:,} / total {tot:,} params")
        
        # 3. Logit scales (unique to each prompted model)
        self.logit_scale_photo = self.clip_photo.logit_scale
        self.logit_scale_sketch = self.clip_sketch.logit_scale

        # -- Visual prompt learners --
        # --use_visual_exchange off (default): independent per-branch prompts,
        # no cross-modal exchange (--vision_depth for both visual branches).
        # --use_visual_exchange on: single shared learner owning both
        # branches, with a fixed directional exchange (see
        # VisualVisualPromptLearner) -- [0, cross_layer) sketch->photo,
        # [cross_layer, vision_depth) photo->sketch, no gate yet.
        self.use_visual_exchange = getattr(cfg, 'use_visual_exchange', False)
        if self.use_visual_exchange:
            self.visual_visual_learner = VisualVisualPromptLearner(cfg, self.clip_photo, self.clip_sketch)
        else:
            self.visual_prompt_photo = VisualPromptLearner(cfg, self.clip_photo)
            self.visual_prompt_sketch = VisualPromptLearner(cfg, self.clip_sketch)

        cfg_photo = copy.copy(cfg)
        cfg_photo.ctx_init = getattr(cfg, 'ctx_init', 'a photo of a')
        self.text_prompt_photo = TextPromptLearner(cfg_photo, classnames, self.clip_photo)

        cfg_sketch = copy.copy(cfg)
        cfg_sketch.ctx_init = getattr(cfg, 'ctx_init_sketch', 'a sketch of a')
        self.text_prompt_sketch = TextPromptLearner(cfg_sketch, classnames, self.clip_sketch)

        # -- Encoders (feed prompts into CLIP's per-layer injection) --
        self.text_encoder_photo = TextEncoder(self.clip_photo)
        self.text_encoder_sketch = TextEncoder(self.clip_sketch)
        self.visual_encoder_photo = VisualEncoder(self.clip_photo)
        self.visual_encoder_sketch = VisualEncoder(self.clip_sketch)

    def normalize_features(self, feat_prenorm):
        """L2-normalize feature tensors."""
        return feat_prenorm / feat_prenorm.norm(dim=-1, keepdim=True)

    def forward(self, x, classnames):
        if len(x) == 5:
            sk_tensor, photo_tensor, neg_tensor, label, filename = x
        elif len(x) == 7:
            sk_tensor, photo_tensor, neg_tensor, _, _, label, filename = x
        else:
            sk_tensor, photo_tensor, neg_tensor, _, _, label = x[:6]

        # Text branch: text features for ALL classes (needed for classification logits)
        text_input_photo, deeper_text_photo = self.text_prompt_photo()
        text_feat_photo_raw = self.text_encoder_photo(
            text_input_photo, self.text_prompt_photo.tokenized_prompts, deeper_text_photo
        )

        text_input_sketch, deeper_text_sketch = self.text_prompt_sketch()
        text_feat_sketch_raw = self.text_encoder_sketch(
            text_input_sketch, self.text_prompt_sketch.tokenized_prompts, deeper_text_sketch
        )

        # Visual branch
        if self.use_visual_exchange:
            vis_shallow_photo, vis_shallow_sketch, vis_deeper_photo, vis_deeper_sketch = self.visual_visual_learner()
        else:
            vis_shallow_photo, vis_deeper_photo = self.visual_prompt_photo()
            vis_shallow_sketch, vis_deeper_sketch = self.visual_prompt_sketch()

        photo_feat_raw = self.visual_encoder_photo(photo_tensor.type(self.dtype), vis_shallow_photo, vis_deeper_photo)
        neg_feat_raw = self.visual_encoder_photo(neg_tensor.type(self.dtype), vis_shallow_photo, vis_deeper_photo)
        sketch_feat_raw = self.visual_encoder_sketch(sk_tensor.type(self.dtype), vis_shallow_sketch, vis_deeper_sketch)

        photo_feat = photo_feat_raw / photo_feat_raw.norm(dim=-1, keepdim=True)
        sketch_feat = sketch_feat_raw / sketch_feat_raw.norm(dim=-1, keepdim=True)
        neg_feat = neg_feat_raw / neg_feat_raw.norm(dim=-1, keepdim=True)
        text_feat_photo = text_feat_photo_raw / text_feat_photo_raw.norm(dim=-1, keepdim=True)
        text_feat_sketch = text_feat_sketch_raw / text_feat_sketch_raw.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale_photo.exp()
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

    def configure_optimizers(self):
        def add_unique_params(candidates, out_list, seen_ids):
            for p in candidates:
                if p.requires_grad and id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    out_list.append(p)

        seen_ids = set()

        prompt_params = []
        if self.model.use_visual_exchange:
            add_unique_params(self.model.visual_visual_learner.parameters(), prompt_params, seen_ids)
        else:
            add_unique_params(self.model.visual_prompt_photo.parameters(), prompt_params, seen_ids)
            add_unique_params(self.model.visual_prompt_sketch.parameters(), prompt_params, seen_ids)
        add_unique_params(self.model.text_prompt_photo.parameters(), prompt_params, seen_ids)
        add_unique_params(self.model.text_prompt_sketch.parameters(), prompt_params, seen_ids)

        non_prompt_params = []
        for _, p in self.model.named_parameters():
            if p.requires_grad and id(p) not in seen_ids:
                seen_ids.add(id(p))
                non_prompt_params.append(p)

        self.print(f"Number of trainable prompt params: {sum(p.numel() for p in prompt_params):,}")
        self.print(f"Number of trainable non-prompt params: {sum(p.numel() for p in non_prompt_params):,}")

        prompt_lr = getattr(self.cfg, 'prompt_lr', 1e-5)
        clip_ln_lr = getattr(self.cfg, 'clip_LN_lr', 1e-5)
        weight_decay = getattr(self.cfg, 'weight_decay', 1e-4)

        param_groups = [{'params': prompt_params, 'lr': prompt_lr}]
        if non_prompt_params:
            param_groups.append({'params': non_prompt_params, 'lr': clip_ln_lr})

        return torch.optim.Adam(param_groups, weight_decay=weight_decay)

    def training_step(self, batch, batch_idx):
        from src.losses_hicropl import loss_fn_hicropl
        features = self.model(batch, self.classnames)
        loss = loss_fn_hicropl(self.args, features)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)

        return loss

    def extract_eval_features(self, tensor, modality):
        """Extract visual features from the prompted encoder for the given modality."""
        if self.model.use_visual_exchange:
            vis_shallow_photo, vis_shallow_sketch, vis_deeper_photo, vis_deeper_sketch = self.model.visual_visual_learner()
            if modality == 'photo':
                vis_shallow, vis_deeper = vis_shallow_photo, vis_deeper_photo
                visual_encoder = self.model.visual_encoder_photo
            else:
                vis_shallow, vis_deeper = vis_shallow_sketch, vis_deeper_sketch
                visual_encoder = self.model.visual_encoder_sketch
        else:
            if modality == 'photo':
                visual_prompt = self.model.visual_prompt_photo
                visual_encoder = self.model.visual_encoder_photo
            else:
                visual_prompt = self.model.visual_prompt_sketch
                visual_encoder = self.model.visual_encoder_sketch
            vis_shallow, vis_deeper = visual_prompt()

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
