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
    VisualEncoder,
    VisualVisualPromptLearner,
    IndependentVisualPromptLearner,
)
from src.clip import clip as _clip


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

        original_device = next(clip_model.parameters()).device
        self.dtype = clip_model.dtype

        # 1. Branch-specific models (3 deep copies + 2 distill branches)
        self.clip_photo = copy.deepcopy(clip_model).to(original_device)
        self.clip_sketch = copy.deepcopy(clip_model).to(original_device)
        self.clip_distill_photo = copy.deepcopy(clip_model_frozen).to(original_device)
        self.clip_distill_sketch = copy.deepcopy(clip_model_frozen).to(original_device)

        # Backward-compatible alias for older code paths
        self.clip_distill = self.clip_distill_photo

        self.clip_sketch.apply(freeze_all_but_bn)
        self.clip_photo.apply(freeze_all_but_bn)
        self.clip_distill_photo.apply(freeze_all_but_bn)  
        self.clip_distill_sketch.apply(freeze_all_but_bn) 

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
            ("clip_distill_photo", self.clip_distill_photo),
            ("clip_distill_sketch", self.clip_distill_sketch),
        ):
            tot, tr = _count_trainable(module)
            print(f"{name}: trainable {tr:,} / total {tot:,} params")
        
        # 3. Logit scales (unique to each prompted model)
        self.logit_scale_photo = self.clip_photo.logit_scale
        self.logit_scale_sketch = self.clip_sketch.logit_scale

        # -- Prompt Learners --
        if getattr(cfg, 'wo_cross_domain', False):
            print("Ablation wo_cross_domain: Initializing Independent Prompt Learner (no cross-domain flow)...")
            self.visual_visual_learner = IndependentVisualPromptLearner(cfg, self.clip_sketch, self.clip_photo)
        else:
            print("Initializing Visual-Visual Prompt Learner (sketch <-> photo)...")
            self.visual_visual_learner = VisualVisualPromptLearner(cfg, self.clip_sketch, self.clip_photo)

        # -- Text encoder: frozen (chỉ LN trainable) + shallow ctx per modality --
        text_src = self.clip_distill_photo
        self.text_transformer = text_src.transformer
        self.text_ln_final    = text_src.ln_final
        self.text_transformer.apply(freeze_all_but_bn)
        self.register_buffer("text_pos_embed", text_src.positional_embedding.data.clone())
        self.register_buffer("text_proj",      text_src.text_projection.data.clone())

        prefix_photo  = getattr(cfg, "ctx_init",        "a photo of a")
        prefix_sketch = getattr(cfg, "ctx_init_sketch", "a sketch of a")
        n_ctx    = int(getattr(cfg, 'n_ctx', 4))
        ctx_dim  = text_src.ln_final.weight.shape[0]   # 512

        with torch.no_grad():
            ph_prefix_tok = _clip.tokenize([prefix_photo]).to(original_device)
            sk_prefix_tok = _clip.tokenize([prefix_sketch]).to(original_device)
            ph_prefix_emb = text_src.token_embedding(ph_prefix_tok).type(self.dtype)
            sk_prefix_emb = text_src.token_embedding(sk_prefix_tok).type(self.dtype)
            ctx_photo_init  = ph_prefix_emb[0, 1:1 + n_ctx, :].clone()
            ctx_sketch_init = sk_prefix_emb[0, 1:1 + n_ctx, :].clone()

        self.ctx_photo  = nn.Parameter(ctx_photo_init)
        self.ctx_sketch = nn.Parameter(ctx_sketch_init)
        self.dropout_ctx = nn.Dropout(p=0.1)

        eot_id = int(_clip.tokenize([""])[0, 1].item())
        ph_prefix_len = int((ph_prefix_tok[0] == eot_id).nonzero()[0].item())
        sk_prefix_len = int((sk_prefix_tok[0] == eot_id).nonzero()[0].item())

        placeholder = " ".join(["X"] * n_ctx)
        ph_tmpl = [f"{prefix_photo} {placeholder} {n}.".replace("_", " ") for n in classnames]
        sk_tmpl = [f"{prefix_sketch} {placeholder} {n}.".replace("_", " ") for n in classnames]

        with torch.no_grad():
            ph_tok = _clip.tokenize(ph_tmpl).to(original_device)
            sk_tok = _clip.tokenize(sk_tmpl).to(original_device)
            ph_emb = text_src.token_embedding(ph_tok).type(self.dtype)
            sk_emb = text_src.token_embedding(sk_tok).type(self.dtype)

        self.register_buffer("text_prefix_photo",  ph_emb[:, :ph_prefix_len, :])
        self.register_buffer("text_suffix_photo",  ph_emb[:, ph_prefix_len + n_ctx:, :])
        self.register_buffer("text_prefix_sketch", sk_emb[:, :sk_prefix_len, :])
        self.register_buffer("text_suffix_sketch", sk_emb[:, sk_prefix_len + n_ctx:, :])
        self.register_buffer("text_tok_photo",  ph_tok)
        self.register_buffer("text_tok_sketch", sk_tok)

        print(f"Text ctx: n_ctx={n_ctx}, ctx_dim={ctx_dim}, "
              f"ph_prefix_len={ph_prefix_len}, sk_prefix_len={sk_prefix_len}")

        # -- Visual Encoders --
        self.visual_encoder_photo = VisualEncoder(self.clip_photo)
        self.visual_encoder_sketch = VisualEncoder(self.clip_sketch)

        gpt_text_file = getattr(cfg, 'gpt_text_file', 'gpt_file/sketchy_ext.json')
        gpt_prompts = _load_gpt_distill_prompts(classnames, gpt_text_file)
        if classnames:
            self.register_buffer("tokenized_gpt_photo", _clip.tokenize(gpt_prompts["photo"], truncate=True))
            self.register_buffer("tokenized_gpt_sketch", _clip.tokenize(gpt_prompts["sketch"], truncate=True))
        else:
            self.register_buffer("tokenized_gpt_photo", torch.empty(0, 77, dtype=torch.long))
            self.register_buffer("tokenized_gpt_sketch", torch.empty(0, 77, dtype=torch.long))

        # -- Extractors removed: logic will be inlined in forward() --

    def normalize_features(self, feat_prenorm):
        """L2-normalize feature tensors."""
        return feat_prenorm / feat_prenorm.norm(dim=-1, keepdim=True)

    def encode_text(self, modality):
        """Shallow CoOp-style text encoding. Frozen transformer, only LN + ctx trainable."""
        if modality == "photo":
            ctx, prefix, suffix, tok = (
                self.ctx_photo, self.text_prefix_photo,
                self.text_suffix_photo, self.text_tok_photo,
            )
        else:
            ctx, prefix, suffix, tok = (
                self.ctx_sketch, self.text_prefix_sketch,
                self.text_suffix_sketch, self.text_tok_sketch,
            )
        if self.training:
            ctx = self.dropout_ctx(ctx)
        n_cls = prefix.shape[0]
        ctx_exp = ctx.unsqueeze(0).expand(n_cls, -1, -1)
        prompts = torch.cat([prefix, ctx_exp, suffix], dim=1)
        x = prompts + self.text_pos_embed.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.text_transformer(x)
        x = x.permute(1, 0, 2)
        x = self.text_ln_final(x).type(self.dtype)
        x = x[torch.arange(n_cls), tok.argmax(dim=-1)] @ self.text_proj
        return x / x.norm(dim=-1, keepdim=True)

    def forward(self, x, classnames):
        """
        Forward pass for training with optimized redundancy.
        Calls visual learner ONCE and routes prompts by branch.
        """
        sk_tensor, photo_tensor, neg_tensor, sk_aug_tensor, photo_aug_tensor, label = x[:6]

        # 1. Call visual-visual learner ONCE (shared by both branches)
        vis1_shallow, vis2_shallow, vis1_deeper, vis2_deeper = self.visual_visual_learner()

        # 2. Photo branch
        text_features_all_photo = self.encode_text("photo")
        image_features_photo = self.visual_encoder_photo(photo_tensor.type(self.dtype), vis2_shallow, vis2_deeper)
        out_p = {
            "image_features": image_features_photo,
            "text_features": text_features_all_photo,
            "text_features_all": text_features_all_photo,
            "logit_scale": self.logit_scale_photo.exp()
        }

        # 3. Sketch branch
        text_features_all_sketch = self.encode_text("sketch")
        image_features_sketch = self.visual_encoder_sketch(sk_tensor.type(self.dtype), vis1_shallow, vis1_deeper)
        out_s = {
            "image_features": image_features_sketch,
            "text_features": text_features_all_sketch,
            "text_features_all": text_features_all_sketch,
            "logit_scale": self.logit_scale_sketch.exp()
        }
        
        # 4. Negative branch (uses photo encoder + photo visual prompts)
        image_features_neg = self.visual_encoder_photo(neg_tensor.type(self.dtype), vis2_shallow, vis2_deeper)
        out_neg = {
            "image_features": image_features_neg,
            "text_features": text_features_all_photo,
            "text_features_all": text_features_all_photo,
            "logit_scale": self.logit_scale_photo.exp()
        }
        
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

        # 3. Residual Mix & Final Normalization
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

        text_feat_photo_prompted = out_p["text_features"]
        text_feat_photo = text_feat_photo_prompted / text_feat_photo_prompted.norm(dim=-1, keepdim=True)

        text_feat_sketch_prompted = out_s["text_features"]
        text_feat_sketch = text_feat_sketch_prompted / text_feat_sketch_prompted.norm(dim=-1, keepdim=True)

        # Encode GPT distill features for all classes (loss will select batch entries)
        text_distill_photo = self.clip_distill_photo.encode_text(self.tokenized_gpt_photo)
        text_distill_photo = text_distill_photo / text_distill_photo.norm(dim=-1, keepdim=True)

        text_distill_sketch = self.clip_distill_sketch.encode_text(self.tokenized_gpt_sketch)
        text_distill_sketch = text_distill_sketch / text_distill_sketch.norm(dim=-1, keepdim=True)

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
            text_distill_photo, text_distill_sketch,
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
            tokens_text_photo  = self.model.ctx_photo.shape[0]
            tokens_text_sketch = self.model.ctx_sketch.shape[0]
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
        # Visual-visual learner + shallow text ctx
        add_unique_params(self.model.visual_visual_learner.parameters(), prompt_params, seen_ids)
        add_unique_params([self.model.ctx_photo, self.model.ctx_sketch], prompt_params, seen_ids)

        ln_params = []
        # LN từ clip encoders + frozen text transformer (chỉ LN trainable) + text_ln_final
        learner_modules = {'visual_visual_learner'}
        add_unique_params(self.model.text_transformer.parameters(), ln_params, seen_ids)
        add_unique_params(self.model.text_ln_final.parameters(),    ln_params, seen_ids)
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                if not any(lm in name for lm in learner_modules):
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
        vis1_shallow, vis2_shallow, vis1_deeper, vis2_deeper = self.model.visual_visual_learner()

        if modality == 'photo':
            visual_encoder = self.model.visual_encoder_photo
            distill_encoder = self.model.clip_distill_photo.visual
            vis_shallow, vis_deeper = vis2_shallow, vis2_deeper
        else:
            visual_encoder = self.model.visual_encoder_sketch
            distill_encoder = self.model.clip_distill_sketch.visual
            vis_shallow, vis_deeper = vis1_shallow, vis1_deeper

        prompted_feat = visual_encoder(tensor.type(self.model.dtype), vis_shallow, vis_deeper)
        prompted_feat_norm = prompted_feat / prompted_feat.norm(dim=-1, keepdim=True)
        
        fixed_feat = distill_encoder(tensor.type(self.model.dtype))
        fixed_feat_norm = fixed_feat / fixed_feat.norm(dim=-1, keepdim=True)
        
        combined_prenorm = prompted_feat_norm + fixed_feat_norm
        return combined_prenorm / combined_prenorm.norm(dim=-1, keepdim=True)

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
