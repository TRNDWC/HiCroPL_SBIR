import copy
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision

from src.hicropl import (
    CrossModalPromptLearner,
    TextEncoder,
    VisualEncoder,
)
from src.hicropl_extractor import HiCroPLFeatureExtractor


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

def freeze_all_but_ln_last_k_layers(model: nn.Module, k: int):
    """
    Freeze toàn bộ model trước, sau đó chỉ mở LayerNorm của k layer cuối.

    Semantics:
    - k <= 0: keep everything frozen
    - 0 < k < num_layers: only LayerNorms in the last-k blocks are trainable
    - k >= num_layers: equivalent to applying freeze_all_but_bn to the whole model

    This helper is intentionally conservative: it never leaves non-LayerNorm
    parameters trainable inside the selected blocks.
    """

    def _freeze_all_params(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad_(False)

    def _freeze_block_params(block: nn.Module) -> None:
        for p in block.parameters():
            p.requires_grad_(False)

    def _get_resblocks(transformer_module: nn.Module):
        resblocks = getattr(transformer_module, "resblocks", None)
        if resblocks is None:
            return []
        try:
            return list(resblocks)
        except TypeError:
            return []

    def _count_blocks() -> tuple[int, int]:
        visual = getattr(model, 'visual', None)
        vis_blocks = []
        if visual is not None:
            vis_blocks = _get_resblocks(getattr(visual, 'transformer', None))

        txt_blocks = _get_resblocks(getattr(model, 'transformer', None))
        return len(vis_blocks), len(txt_blocks)

    n_vis_blocks, n_txt_blocks = _count_blocks()

    if k <= 0:
        _freeze_all_params(model)
        return

    # Start from the exact same baseline as `freeze_all_but_bn` on the full model.
    # Then we selectively freeze back the blocks that are NOT in the last-k window.
    model.apply(freeze_all_but_bn)

    # ===================== VISUAL ENCODER =====================
    visual = getattr(model, 'visual', None)
    if visual is not None:
        resblocks = _get_resblocks(getattr(visual, 'transformer', None))
        if resblocks:
            blocks = list(resblocks)
            num_blocks = len(blocks)
            freeze_count = max(0, num_blocks - k)

            # Freeze back the blocks that are outside the last-k window.
            for block in blocks[:freeze_count]:
                _freeze_block_params(block)

        # Keep common visual LNs exactly as the full-model baseline.
        # No extra action needed for the last-k blocks.

    # ===================== TEXT ENCODER =====================
    resblocks = _get_resblocks(getattr(model, 'transformer', None))
    if resblocks is not None:
        blocks = list(resblocks)
        num_blocks = len(blocks)
        freeze_count = max(0, num_blocks - k)

        # Freeze back the blocks that are outside the last-k window.
        for block in blocks[:freeze_count]:
            _freeze_block_params(block)

def _normalize_classname(name):
    return str(name).strip().lower().replace(" ", "_")


def _resolve_text_file(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def _load_gpt_distill_prompts(classnames, gpt_text_file):
    text_file = _resolve_text_file(gpt_text_file)
    if not text_file.exists():
        raise FileNotFoundError(f"GPT text file not found: {text_file}")

    with text_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    prompts_by_modality = {"photo": {}, "sketch": {}}
    for row in rows:
        cls = _normalize_classname(row.get("class", ""))
        input_text = str(row.get("input", "")).lower()
        output_text = str(row.get("output", "")).strip()
        if not cls or not output_text:
            continue
        if "sketch" in input_text:
            prompts_by_modality["sketch"][cls] = output_text
        elif "photo" in input_text:
            prompts_by_modality["photo"][cls] = output_text

    prompts = {"photo": [], "sketch": []}
    missing = {"photo": [], "sketch": []}
    for classname in classnames:
        key = _normalize_classname(classname)
        for modality in ("photo", "sketch"):
            prompt = prompts_by_modality[modality].get(key)
            if prompt is None:
                missing[modality].append(classname)
                prompt = f"a {modality} of a {str(classname).replace('_', ' ')}."
            prompts[modality].append(prompt)

    for modality, names in missing.items():
        if names:
            print(
                f"Warning: missing {len(names)} {modality} GPT prompts in {text_file}; "
                "falling back to template prompts."
            )

    return prompts


class CustomCLIP(nn.Module):
    """
    HiCroPL-SBIR Architecture Wrapper.
    Sử dụng HiCroPLFeatureExtractor làm nòng cốt.
    """

    def __init__(self, cfg, clip_model, clip_model_frozen, classnames=None):
        super().__init__()
        self.cfg = cfg
        
        if classnames is None:
            classnames = []
        if len(classnames) == 0:
            raise ValueError("CustomCLIP requires non-empty classnames during initialization.")

        num_trainable_ln = getattr(cfg, 'num_trainable_ln', -1)
        original_device = next(clip_model.parameters()).device
        self.dtype = clip_model.dtype

        # 1. Branch-specific models (4 deep copies for independent training)
        self.clip_photo = copy.deepcopy(clip_model).to(original_device)
        self.clip_sketch = copy.deepcopy(clip_model).to(original_device)
        self.clip_distill_photo = copy.deepcopy(clip_model_frozen).to(original_device)
        self.clip_distill_sketch = copy.deepcopy(clip_model_frozen).to(original_device)

        # 2. Set Trainable LayerNorms
        self.clip_photo.apply(freeze_all_but_bn)
        self.clip_sketch.apply(freeze_all_but_bn)
        
        # Distill/Augment Branches:
        freeze_all_but_ln_last_k_layers(self.clip_distill_photo, num_trainable_ln)
        freeze_all_but_ln_last_k_layers(self.clip_distill_sketch, num_trainable_ln)
        
        # 3. Logit scales (unique to each prompted model)
        self.logit_scale_photo = self.clip_photo.logit_scale
        self.logit_scale_sketch = self.clip_sketch.logit_scale

        # -- Prompt Learners --
        print("Initializing Photo Prompt Learner...")
        cfg_photo = copy.copy(cfg)
        cfg_photo.ctx_init = getattr(cfg, 'ctx_init', 'a photo of a')
        self.prompt_learner_photo = CrossModalPromptLearner(
            cfg=cfg_photo,
            classnames=classnames,
            clip_model=self.clip_photo,
            clip_model_distill=self.clip_distill_photo
        )

        print("Initializing Sketch Prompt Learner...")
        cfg_sketch = copy.copy(cfg)
        cfg_sketch.ctx_init = getattr(cfg, 'ctx_init_sketch', 'a sketch of a')
        self.prompt_learner_sketch = CrossModalPromptLearner(
            cfg=cfg_sketch,
            classnames=classnames,
            clip_model=self.clip_sketch,
            clip_model_distill=self.clip_distill_sketch
        )

        # -- Encoders (Main Branches using their own models with ALL LNs open) --
        self.text_encoder_photo = TextEncoder(self.clip_photo)
        self.text_encoder_sketch = TextEncoder(self.clip_sketch)
        self.visual_encoder_photo = VisualEncoder(self.clip_photo)
        self.visual_encoder_sketch = VisualEncoder(self.clip_sketch)

        # -- GPT Text Distill Tokenization (TEMPORARILY DISABLED) --
        # gpt_text_file = getattr(cfg, 'gpt_text_file', 'gpt_file/sketchy_ext.json')
        # gpt_prompts = _load_gpt_distill_prompts(classnames, gpt_text_file)
        # from src.clip import clip as _clip
        # if classnames:
        #     self.register_buffer("tokenized_gpt_photo", _clip.tokenize(gpt_prompts["photo"], truncate=True))
        #     self.register_buffer("tokenized_gpt_sketch", _clip.tokenize(gpt_prompts["sketch"], truncate=True))
        # else:
        #     self.register_buffer("tokenized_gpt_photo", torch.empty(0, 77, dtype=torch.long))
        #     self.register_buffer("tokenized_gpt_sketch", torch.empty(0, 77, dtype=torch.long))
        
        # -- HiCroPL Extractors (Main Branches) --
        self.extractor_photo = HiCroPLFeatureExtractor(
            prompt_learner=self.prompt_learner_photo,
            text_encoder=self.text_encoder_photo,
            image_encoder=self.visual_encoder_photo,
            logit_scale=self.logit_scale_photo,
            dtype=self.dtype,
        )
        self.extractor_sketch = HiCroPLFeatureExtractor(
            prompt_learner=self.prompt_learner_sketch,
            text_encoder=self.text_encoder_sketch,
            image_encoder=self.visual_encoder_sketch,
            logit_scale=self.logit_scale_sketch,
            dtype=self.dtype,
        )

    def normalize_features(self, feat_prenorm):
        """L2-normalize feature tensors."""
        return feat_prenorm / feat_prenorm.norm(dim=-1, keepdim=True)

    def forward(self, x, classnames):
        """
        Forward pass for training with optimized redundancy.
        """
        sk_tensor, photo_tensor, neg_tensor, sk_aug_tensor, photo_aug_tensor, label = x[:6]
        
        # 1. Prompted Features (Main Branches)
        out_p = self.extractor_photo(photo_tensor)
        out_s = self.extractor_sketch(sk_tensor)
        out_neg = self.extractor_photo(neg_tensor)
        
        # 2. Distill Visual Features (Open LN branches) - RUN ONCE
        photo_aug_feat_fixed = self.clip_distill_photo.visual(photo_aug_tensor.type(self.dtype))
        photo_aug_feat_fixed = photo_aug_feat_fixed / photo_aug_feat_fixed.norm(dim=-1, keepdim=True)
        
        sketch_aug_feat_fixed = self.clip_distill_sketch.visual(sk_aug_tensor.type(self.dtype))
        sketch_aug_feat_fixed = sketch_aug_feat_fixed / sketch_aug_feat_fixed.norm(dim=-1, keepdim=True)

        # Distill Visual Features for Original (for residual mix)
        photo_feat_fixed = self.clip_distill_photo.visual(photo_tensor.type(self.dtype))
        photo_feat_fixed = photo_feat_fixed / photo_feat_fixed.norm(dim=-1, keepdim=True)
        
        sketch_feat_fixed = self.clip_distill_sketch.visual(sk_tensor.type(self.dtype))
        sketch_feat_fixed = sketch_feat_fixed / sketch_feat_fixed.norm(dim=-1, keepdim=True)

        # 3. GPT Text Distill Features (TEMPORARILY DISABLED)
        # text_distill_photo = self.clip_distill_photo.encode_text(self.tokenized_gpt_photo)
        # text_distill_photo = text_distill_photo / text_distill_photo.norm(dim=-1, keepdim=True)
        
        # text_distill_sketch = self.clip_distill_sketch.encode_text(self.tokenized_gpt_sketch)
        # text_distill_sketch = text_distill_sketch / text_distill_sketch.norm(dim=-1, keepdim=True)

        # Filter by labels for loss computation
        # text_distill_photo_batch = text_distill_photo[label]
        # text_distill_sketch_batch = text_distill_sketch[label]

        # 4. Residual Mix & Final Normalization
        # Image
        photo_feat_prompted = out_p["image_features"]
        photo_feat_prompted_norm = photo_feat_prompted / photo_feat_prompted.norm(dim=-1, keepdim=True)
        photo_feat_prenorm = photo_feat_prompted_norm + photo_feat_fixed
        photo_feat = photo_feat_prenorm / photo_feat_prenorm.norm(dim=-1, keepdim=True)

        sketch_feat_prompted = out_s["image_features"]
        sketch_feat_prompted_norm = sketch_feat_prompted / sketch_feat_prompted.norm(dim=-1, keepdim=True)
        sketch_feat_prenorm = sketch_feat_prompted_norm + sketch_feat_fixed
        sketch_feat = sketch_feat_prenorm / sketch_feat_prenorm.norm(dim=-1, keepdim=True)
        
        neg_feat_prompted = out_neg["image_features"]
        neg_feat = neg_feat_prompted / neg_feat_prompted.norm(dim=-1, keepdim=True)

        # Text (ONLY PROMPTED - GPT DISABLED)
        text_feat_photo_prompted = out_p["text_features"]
        text_feat_photo = text_feat_photo_prompted / text_feat_photo_prompted.norm(dim=-1, keepdim=True)

        text_feat_sketch_prompted = out_s["text_features"]
        text_feat_sketch = text_feat_sketch_prompted / text_feat_sketch_prompted.norm(dim=-1, keepdim=True)

        # 5. Compute Logits
        logit_scale = out_p["logit_scale"]
        logits_photo = logit_scale * photo_feat @ text_feat_photo.t()
        logits_sketch = logit_scale * sketch_feat @ text_feat_sketch.t()
        
        # Logits for Augmented Images
        logits_photo_aug = logit_scale * photo_aug_feat_fixed @ text_feat_photo.t()
        logits_sketch_aug = logit_scale * sketch_aug_feat_fixed @ text_feat_sketch.t()
        
        return (
            photo_feat, logits_photo,
            sketch_feat, logits_sketch,
            neg_feat, label,
            photo_aug_feat_fixed, sketch_aug_feat_fixed,
            logits_photo_aug, logits_sketch_aug,
            text_feat_photo, text_feat_sketch,
            None, None, # text_distill_photo_batch, text_distill_sketch_batch (DISABLED)
            None, None, # text_distill_photo, text_distill_sketch (DISABLED)
            photo_feat_fixed, sketch_feat_fixed,
        )


class HiCroPL_SBIR(pl.LightningModule):
    def __init__(self, cfg, args, classnames, model):
        super().__init__()
        self.cfg = cfg
        self.args = args
        self.classnames = classnames
        self.model = model
        
        self.best_metric = 1e-3

        self.eval_mode = getattr(args, 'eval_mode', 'category')

        self.test_photo_features = []
        self.test_sketch_features = []
        self.test_photo_labels = []
        self.test_sketch_labels = []

        from collections import defaultdict
        self.fg_sketch_buckets = defaultdict(lambda: {
            'features': [],
            'filenames': [],
            'base_names': []
        })
        self.fg_photo_buckets = defaultdict(lambda: {
            'features': [],
            'filenames': [],
            'base_names': []
        })

    def on_train_epoch_start(self):
        self.model.visual_encoder_photo.eval()
        self.model.visual_encoder_sketch.eval()
        self.model.text_encoder_photo.eval()
        self.model.text_encoder_sketch.eval()
        self.model.clip_photo.eval()
        self.model.clip_sketch.eval()
        self.model.clip_distill_photo.eval()
        self.model.clip_distill_sketch.eval()

    def configure_optimizers(self):
        def add_unique_params(candidates, out_list, seen_ids):
            for p in candidates:
                if p.requires_grad and id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    out_list.append(p)

        seen_ids = set()

        prompt_params = []
        add_unique_params(self.model.prompt_learner_photo.parameters(), prompt_params, seen_ids)
        add_unique_params(self.model.prompt_learner_sketch.parameters(), prompt_params, seen_ids)

        ln_params = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.LayerNorm):
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
        """Extract visual features: Prompted + Distill Fixed (Residual Mix)"""
        if modality == 'photo':
            extractor = self.model.extractor_photo
            distill_encoder = self.model.clip_distill_photo.visual
        else:
            extractor = self.model.extractor_sketch
            distill_encoder = self.model.clip_distill_sketch.visual
            
        out = extractor(tensor)
        prompted_feat = out["image_features"]
        prompted_feat_norm = prompted_feat / prompted_feat.norm(dim=-1, keepdim=True)
        
        fixed_feat = distill_encoder(tensor.type(self.model.dtype))
        fixed_feat_norm = fixed_feat / fixed_feat.norm(dim=-1, keepdim=True)
        
        combined_prenorm = prompted_feat_norm + fixed_feat_norm
        return combined_prenorm / combined_prenorm.norm(dim=-1, keepdim=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.eval_mode == 'fine_grained':
            return self._validation_step_fg(batch, batch_idx, dataloader_idx)
        else:
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

    def _validation_step_fg(self, batch, batch_idx, dataloader_idx=0):
        tensor, category_idx, filename, base_name = batch

        if dataloader_idx == 0:
            sketch_feat = self.extract_eval_features(tensor, modality='sketch')
            target_buckets = self.fg_sketch_buckets
        elif dataloader_idx == 1:
            photo_feat = self.extract_eval_features(tensor, modality='photo')
            target_buckets = self.fg_photo_buckets

        for i in range(tensor.size(0)):
            cat_idx = category_idx[i].item()
            feat = sketch_feat[i] if dataloader_idx == 0 else photo_feat[i]
            fname = filename[i]
            bname = base_name[i]
            target_buckets[cat_idx]['features'].append(feat.cpu().detach())  
            target_buckets[cat_idx]['filenames'].append(fname)
            target_buckets[cat_idx]['base_names'].append(bname)

    def on_validation_epoch_end(self):
        if self.eval_mode == 'fine_grained':
            return self._on_validation_epoch_end_fine_grained()
        else:
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

    def _on_validation_epoch_end_fine_grained(self):
        from src_fg.utils_fg import compute_rank_based_accuracy
        
        if len(self.fg_sketch_buckets) == 0 or len(self.fg_photo_buckets) == 0:
            self.print("Warning: No fine-grained data collected. Skipping FG metrics.")
            return
        
        all_ranks = []
        for category_idx in self.fg_sketch_buckets.keys():
            if category_idx not in self.fg_photo_buckets:
                continue
            
            sketch_bucket = self.fg_sketch_buckets[category_idx]
            photo_bucket = self.fg_photo_buckets[category_idx]
            
            if len(sketch_bucket['features']) == 0 or len(photo_bucket['features']) == 0:
                continue
            
            sketch_feats = torch.stack(sketch_bucket['features'])  
            photo_feats = torch.stack(photo_bucket['features'])    
            
            ranks = self._compute_per_category_rank(
                sketch_feats,
                sketch_bucket['base_names'],
                photo_feats,
                photo_bucket['base_names']
            )
            all_ranks.append(ranks)
        
        if len(all_ranks) == 0:
            self.print("Warning: No valid categories for FG evaluation.")
            return
        
        all_ranks_tensor = torch.cat(all_ranks)  
        result = compute_rank_based_accuracy(all_ranks_tensor, top_k_list=[1, 5, 10])
        
        acc1 = result['acc@1']
        acc5 = result['acc@5']
        acc10 = result['acc@10']

        self.log('fg_acc@1', acc1, on_epoch=True, prog_bar=True)
        self.log('fg_acc@5', acc5, on_epoch=True, prog_bar=True)
        self.log('fg_acc@10', acc10, on_epoch=True, prog_bar=True)
        self.log('top1', acc1, on_epoch=True, prog_bar=True)
        self.log('top5', acc5, on_epoch=True, prog_bar=True)
        
        if self.global_step > 0:
            self.best_metric = max(self.best_metric, acc1)
        self.log('best_fg_acc@1', self.best_metric, on_epoch=True, prog_bar=False)
        
        self.print(f'top1: {acc1:.4f}, top5: {acc5:.4f}, acc@10: {acc10:.4f}, Best: {self.best_metric:.4f}')
        
        self.fg_sketch_buckets.clear()
        self.fg_photo_buckets.clear()

    def _compute_per_category_rank(self, sketch_feats, sketch_base_names, photo_feats, photo_base_names):
        """Vectorized rank computation for fine-grained retrieval"""
        sim_matrix = sketch_feats @ photo_feats.t()  
        
        N_sk = len(sketch_feats)
        ranks = torch.zeros(N_sk, device=sketch_feats.device)
        
        # We still need to find the ground truth index for each sketch.
        # This part is hard to vectorize completely because it depends on string matching.
        # But we can vectorize the rank calculation once gt_idx is known.
        for i in range(N_sk):
            sketch_base = sketch_base_names[i]
            try:
                gt_idx = photo_base_names.index(sketch_base)
                gt_sim = sim_matrix[i, gt_idx]
                # Rank = number of items with similarity >= ground truth similarity
                rank = (sim_matrix[i] >= gt_sim).sum()
                ranks[i] = rank
            except ValueError:
                ranks[i] = len(photo_base_names) + 1
        
        return ranks

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
