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


def freeze_all_but_bn(m):
    """Freeze all parameters except LayerNorm weights/biases."""
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad_(False)


class SimpleTextPromptLearner(nn.Module):
    """CoOp-style text prompt learner with shallow context tokens only."""

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_ctx = int(getattr(cfg, "n_ctx", 0))
        ctx_init = getattr(cfg, "ctx_init", "a photo of a")
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init and n_ctx > 0:
            prompt = _clip.tokenize(ctx_init.replace("_", " "))
            prompt = prompt.to(clip_model.token_embedding.weight.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init.replace("_", " ")
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            if n_ctx > 0:
                nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx) if n_ctx > 0 else ""

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [f"{prompt_prefix} {name}.".strip() for name in classnames]
        tokenized_prompts = torch.cat([_clip.tokenize(p) for p in prompts]).to(
            clip_model.token_embedding.weight.device
        )
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def forward(self, label=None):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.tokenized_prompts.shape[0], -1, -1)

        if label is not None:
            prefix = self.token_prefix[label]
            suffix = self.token_suffix[label]
            ctx = ctx[label]
        else:
            prefix = self.token_prefix
            suffix = self.token_suffix

        return torch.cat([prefix, ctx, suffix], dim=1)


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

        # Trainable LayerNorms
        self.clip.apply(freeze_model)
        if hasattr(self.clip.visual, "VPT"):
            self.clip.visual.VPT.requires_grad_(False)

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

        # Text prompt learners (shallow only)
        cfg_photo = copy.copy(cfg)
        cfg_photo.ctx_init = getattr(cfg, "ctx_init", "a photo of a")
        cfg_photo.prompt_depth = 1
        self.text_prompt_photo = SimpleTextPromptLearner(cfg_photo, classnames, self.clip)

        cfg_sketch = copy.copy(cfg)
        cfg_sketch.ctx_init = getattr(cfg, "ctx_init_sketch", "a sketch of a")
        cfg_sketch.prompt_depth = 1
        self.text_prompt_sketch = SimpleTextPromptLearner(cfg_sketch, classnames, self.clip)

        # Visual prompts (shallow only), separate for photo and sketch
        prompt_dim = self.clip.visual.conv1.weight.shape[0]
        n_ctx = int(getattr(cfg, "n_ctx", 0))
        self.visual_prompt_photo = nn.Parameter(torch.empty(n_ctx, prompt_dim, dtype=self.dtype))
        self.visual_prompt_sketch = nn.Parameter(torch.empty(n_ctx, prompt_dim, dtype=self.dtype))
        if n_ctx > 0:
            nn.init.normal_(self.visual_prompt_photo, std=0.02)
            nn.init.normal_(self.visual_prompt_sketch, std=0.02)

    def _encode_text_vanilla(self, clip_model, prompts, tokenized_prompts):
        x = prompts + clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ clip_model.text_projection
        return x

    def _encode_image_with_prompt(self, clip_model, image, visual_prompt):
        visual = clip_model.visual
        x = visual.conv1(image.type(self.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = x + visual.positional_embedding.to(x.dtype)

        if visual_prompt is not None and visual_prompt.numel() > 0:
            visual_ctx = visual_prompt.to(device=x.device, dtype=x.dtype)
            visual_ctx = visual_ctx.expand(x.shape[0], -1, -1)
            # Match the legacy VPT-style layout: [CLS] [prompt] [patches]
            x = torch.cat([x[:, :1, :], visual_ctx, x[:, 1:, :]], dim=1)

        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)

        x = visual.ln_post(x[:, 0, :])
        if visual.proj is not None:
            x = x @ visual.proj
        return x

    def forward(self, x, classnames):
        sk_tensor, photo_tensor, neg_tensor, sk_aug_tensor, photo_aug_tensor, label = x[:6]

        text_input_photo_all = self.text_prompt_photo(label=None)  # All classes
        text_features_all_photo = self._encode_text_vanilla(
            self.clip, text_input_photo_all, self.text_prompt_photo.tokenized_prompts
        )
        image_features_photo = self._encode_image_with_prompt(
            self.clip, photo_tensor, self.visual_prompt_photo
        )
        out_p = {
            "image_features": image_features_photo,
            "text_features": text_features_all_photo,
            "text_features_all": text_features_all_photo,
            "logit_scale": self.logit_scale.exp(),
        }

        text_input_sketch_all = self.text_prompt_sketch(label=None)  # All classes
        text_features_all_sketch = self._encode_text_vanilla(
            self.clip, text_input_sketch_all, self.text_prompt_sketch.tokenized_prompts
        )
        image_features_sketch = self._encode_image_with_prompt(
            self.clip, sk_tensor, self.visual_prompt_sketch
        )
        out_s = {
            "image_features": image_features_sketch,
            "text_features": text_features_all_sketch,
            "text_features_all": text_features_all_sketch,
            "logit_scale": self.logit_scale.exp(),
        }

        image_features_neg = self._encode_image_with_prompt(
            self.clip, neg_tensor, self.visual_prompt_photo
        )
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
        if hasattr(self.model.text_prompt_photo, "ctx"):
            tokens_text_photo = self.model.text_prompt_photo.ctx.shape[0]
        if hasattr(self.model.text_prompt_sketch, "ctx"):
            tokens_text_sketch = self.model.text_prompt_sketch.ctx.shape[0]

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
        add_unique_params(self.model.text_prompt_photo.parameters(), prompt_params, seen_ids)
        add_unique_params(self.model.text_prompt_sketch.parameters(), prompt_params, seen_ids)
        add_unique_params([self.model.visual_prompt_photo], prompt_params, seen_ids)
        add_unique_params([self.model.visual_prompt_sketch], prompt_params, seen_ids)

        ln_params = []
        # Only collect LayerNorms from clip encoders (NOT from learners, already included above)
        learner_modules = {
            'text_prompt_photo', 'text_prompt_sketch'
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
            feat = self.model._encode_image_with_prompt(
                self.model.clip, tensor, self.model.visual_prompt_photo
            )
        else:
            feat = self.model._encode_image_with_prompt(
                self.model.clip, tensor, self.model.visual_prompt_sketch
            )
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
