import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision

from src.clip import clip as _clip


def freeze_model(m):
    """Freeze all parameters of the given module."""
    for param in m.parameters():
        param.requires_grad_(False)


def freeze_all_but_ln(clip_model):
    """Freeze every parameter in `clip_model`, then unfreeze LayerNorm params only.

    The previous implementation walked modules and only touched `.weight` / `.bias`,
    which silently left `nn.MultiheadAttention.in_proj_weight/bias` and several loose
    Parameters (visual.proj, text_projection, positional/class embeddings, logit_scale)
    trainable — making "LN-only tuning" effectively fine-tune ~30M params.
    """
    for p in clip_model.parameters():
        p.requires_grad_(False)
    for m in clip_model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters(recurse=False):
                p.requires_grad_(True)




class CustomCLIP(nn.Module):
    """Vanilla CLIP backbone with shallow visual + text prompts."""

    def __init__(self, cfg, clip_model, clip_model_frozen, classnames=None):
        super().__init__()
        self.cfg = cfg

        if classnames is None:
            classnames = []
        if len(classnames) == 0:
            raise ValueError("CustomCLIP requires non-empty classnames during initialization.")

        original_device = next(clip_model.parameters()).device
        self.dtype = clip_model.dtype

        # Shared CLIP backbone
        self.clip = copy.deepcopy(clip_model).to(original_device)

        # Trainable LayerNorms only (true LN-only freeze; see freeze_all_but_ln docstring)
        freeze_all_but_ln(self.clip)

        # Print trainable param counts for verification
        def _count_trainable(m):
            total = 0
            trainable = 0
            for p in m.parameters():
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()
            return total, trainable

        tot, tr = _count_trainable(self.clip)
        print(f"clip: trainable {tr:,} / total {tot:,} params")

        self.logit_scale = self.clip.logit_scale

        # Tokenize classnames for standard CLIP text encoder
        ctx_init_photo = getattr(cfg, "ctx_init", "a photo of a")
        ctx_init_sketch = getattr(cfg, "ctx_init_sketch", "a sketch of a")
        
        prompts_photo = [f"{ctx_init_photo} {name}.".replace("_", " ").strip() for name in classnames]
        prompts_sketch = [f"{ctx_init_sketch} {name}.".replace("_", " ").strip() for name in classnames]
        
        self.register_buffer("tokenized_photo", _clip.tokenize(prompts_photo))
        self.register_buffer("tokenized_sketch", _clip.tokenize(prompts_sketch))

        # Visual prompts (shallow only), separate for photo and sketch
        prompt_dim = self.clip.visual.conv1.weight.shape[0]
        n_ctx = int(getattr(cfg, "n_ctx", 0))
        self.visual_prompt_photo = nn.Parameter(torch.empty(n_ctx, prompt_dim, dtype=self.dtype))
        self.visual_prompt_sketch = nn.Parameter(torch.empty(n_ctx, prompt_dim, dtype=self.dtype))
        if n_ctx > 0:
            nn.init.normal_(self.visual_prompt_photo, std=0.02)
            nn.init.normal_(self.visual_prompt_sketch, std=0.02)



    def forward(self, x, classnames):
        sk_tensor, photo_tensor, neg_tensor, sk_aug_tensor, photo_aug_tensor, label = x[:6]

        text_features_all_photo = self.clip.encode_text(self.tokenized_photo)
        vp_photo = self.visual_prompt_photo.expand(photo_tensor.shape[0], -1, -1) if self.visual_prompt_photo.numel() > 0 else None
        image_features_photo = self.clip.encode_image(photo_tensor, prompt=vp_photo)
        
        out_p = {
            "image_features": image_features_photo,
            "text_features": text_features_all_photo,
            "text_features_all": text_features_all_photo,
            "logit_scale": self.logit_scale.exp(),
        }

        text_features_all_sketch = self.clip.encode_text(self.tokenized_sketch)
        vp_sketch = self.visual_prompt_sketch.expand(sk_tensor.shape[0], -1, -1) if self.visual_prompt_sketch.numel() > 0 else None
        image_features_sketch = self.clip.encode_image(sk_tensor, prompt=vp_sketch)
        
        out_s = {
            "image_features": image_features_sketch,
            "text_features": text_features_all_sketch,
            "text_features_all": text_features_all_sketch,
            "logit_scale": self.logit_scale.exp(),
        }

        vp_neg = self.visual_prompt_photo.expand(neg_tensor.shape[0], -1, -1) if self.visual_prompt_photo.numel() > 0 else None
        image_features_neg = self.clip.encode_image(neg_tensor, prompt=vp_neg)
        
        out_neg = {
            "image_features": image_features_neg,
            "text_features": text_features_all_photo,
            "text_features_all": text_features_all_photo,
            "logit_scale": self.logit_scale.exp(),
        }

        # Final Normalization
        photo_feat_prompted = out_p["image_features"]
        photo_feat = photo_feat_prompted / photo_feat_prompted.norm(dim=-1, keepdim=True)

        sketch_feat_prompted = out_s["image_features"]
        sketch_feat = sketch_feat_prompted / sketch_feat_prompted.norm(dim=-1, keepdim=True)

        neg_feat_prompted = out_neg["image_features"]
        neg_feat = neg_feat_prompted / neg_feat_prompted.norm(dim=-1, keepdim=True)

        text_feat_photo_prompted = out_p["text_features"]
        text_feat_photo = text_feat_photo_prompted / text_feat_photo_prompted.norm(dim=-1, keepdim=True)

        text_feat_sketch_prompted = out_s["text_features"]
        text_feat_sketch = text_feat_sketch_prompted / text_feat_sketch_prompted.norm(dim=-1, keepdim=True)

        logit_scale = out_p["logit_scale"]
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
        tokens_visual_photo = 0
        tokens_visual_sketch = 0
        if hasattr(self.model, "visual_prompt_photo"):
            tokens_visual_photo = self.model.visual_prompt_photo.shape[0]
        if hasattr(self.model, "visual_prompt_sketch"):
            tokens_visual_sketch = self.model.visual_prompt_sketch.shape[0]

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
        add_unique_params([self.model.visual_prompt_photo], prompt_params, seen_ids)
        add_unique_params([self.model.visual_prompt_sketch], prompt_params, seen_ids)

        ln_params = []
        for name, module in self.model.named_modules():
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
            vp = self.model.visual_prompt_photo.expand(tensor.shape[0], -1, -1) if self.model.visual_prompt_photo.numel() > 0 else None
            feat = self.model.clip.encode_image(tensor, prompt=vp)
        else:
            vp = self.model.visual_prompt_sketch.expand(tensor.shape[0], -1, -1) if self.model.visual_prompt_sketch.numel() > 0 else None
            feat = self.model.clip.encode_image(tensor, prompt=vp)
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
