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


def freeze_all_but_ln(m):
    """Per-module hook: freeze weight/bias unless the module is a LayerNorm."""
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad_(False)


class CustomCLIP(nn.Module):
    """CLIP-AT baseline for category-level ZS-SBIR (Sain et al. CVPR'23).

    Key design points (paper §4):
      - Two separate visual encoders F_s, F_p, both initialised from CLIP image encoder.
      - Trainable parameter set: {v_s, v_p, l^s_theta, l^p_theta}
        (per-modality shallow visual prompts + per-modality LayerNorms).
      - Text encoder fully frozen; classification uses a single hard template
        "a photo of a [CLS]" applied to both modalities.
      - Shallow prompt: K=3 tokens injected only in the first transformer layer.
    """

    def __init__(self, cfg, clip_model, clip_model_frozen=None, classnames=None):
        super().__init__()
        self.cfg = cfg

        if classnames is None or len(classnames) == 0:
            raise ValueError("CustomCLIP requires non-empty classnames during initialization.")

        original_device = next(clip_model.parameters()).device
        self.dtype = clip_model.dtype

        # Full CLIP holds the text encoder (+ unused visual; we use visual_sketch/photo instead).
        # Unfreeze only LayerNorms (mirrors what we do for the visual branches).
        self.clip = copy.deepcopy(clip_model).to(original_device)
        freeze_model(self.clip)
        for m in self.clip.transformer.modules():
            if isinstance(m, torch.nn.LayerNorm):
                for p in m.parameters(recurse=False):
                    p.requires_grad_(True)
        for p in self.clip.ln_final.parameters(recurse=False):
            p.requires_grad_(True)

        # Two separate visual encoders (paper: F_s, F_p both init from CLIP visual).
        self.visual_sketch = copy.deepcopy(clip_model.visual).to(original_device)
        self.visual_photo = copy.deepcopy(clip_model.visual).to(original_device)

        # Unfreeze LayerNorms only in the two visual branches (l^s_theta, l^p_theta).
        self.visual_sketch.apply(freeze_all_but_ln)
        self.visual_photo.apply(freeze_all_but_ln)

        # Trainable-param count for sanity logging
        def _count_trainable(m):
            total = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable

        v_tot_s, v_tr_s = _count_trainable(self.visual_sketch)
        v_tot_p, v_tr_p = _count_trainable(self.visual_photo)
        t_tot, t_tr = _count_trainable(self.clip)
        print(f"visual_sketch: trainable {v_tr_s:,} / total {v_tot_s:,}")
        print(f"visual_photo : trainable {v_tr_p:,} / total {v_tot_p:,}")
        print(f"clip (text-branch LN trainable): trainable {t_tr:,} / total {t_tot:,}")

        self.logit_scale = self.clip.logit_scale

        # Tokenize a shared hard template. The first n_ctx context tokens after [SOS]
        # will be REPLACED at forward time by learnable per-modality prompts (CoOp style),
        # so the literal string here is only a structural template.
        ctx_init = getattr(cfg, "ctx_init", "a photo of a")
        prompts = [f"{ctx_init} {name}.".replace("_", " ").strip() for name in classnames]
        self.register_buffer("tokenized_text", _clip.tokenize(prompts).to(original_device))

        # Two separate shallow visual prompts. Paper: K = 3 tokens, dim = 768.
        prompt_dim_v = self.visual_sketch.conv1.weight.shape[0]
        n_ctx = int(getattr(cfg, "n_ctx", 3))
        self.n_ctx = n_ctx
        self.visual_prompt_sketch = nn.Parameter(torch.empty(n_ctx, prompt_dim_v, dtype=self.dtype))
        self.visual_prompt_photo = nn.Parameter(torch.empty(n_ctx, prompt_dim_v, dtype=self.dtype))
        if n_ctx > 0:
            nn.init.normal_(self.visual_prompt_sketch, std=0.02)
            nn.init.normal_(self.visual_prompt_photo, std=0.02)

        # Two separate learnable text prompts, CoOp-style, injected at the first text-encoder layer.
        # They REPLACE the n_ctx context tokens right after [SOS]; class-name suffix + [EOT] + pad
        # are held as frozen buffers.
        prompt_dim_t = self.clip.ln_final.weight.shape[0]
        self.text_prompt_sketch = nn.Parameter(torch.empty(n_ctx, prompt_dim_t, dtype=self.dtype))
        self.text_prompt_photo = nn.Parameter(torch.empty(n_ctx, prompt_dim_t, dtype=self.dtype))
        if n_ctx > 0:
            nn.init.normal_(self.text_prompt_sketch, std=0.02)
            nn.init.normal_(self.text_prompt_photo, std=0.02)

        # Cache the frozen [SOS] prefix and [class + EOT + pad] suffix embeddings.
        with torch.no_grad():
            full_embed = self.clip.token_embedding(self.tokenized_text).type(self.dtype)
        # full_embed: (n_cls, 77, d_t). prefix = token 0 ([SOS]); suffix = tokens 1+n_ctx :
        self.register_buffer("token_prefix", full_embed[:, :1, :])              # (n_cls, 1, d_t)
        self.register_buffer("token_suffix", full_embed[:, 1 + n_ctx:, :])      # (n_cls, 77-1-n_ctx, d_t)

    def encode_visual(self, x, modality):
        """Encode image through the modality-specific visual branch with its visual prompt."""
        if modality == "sketch":
            encoder = self.visual_sketch
            vp = self.visual_prompt_sketch
        else:
            encoder = self.visual_photo
            vp = self.visual_prompt_photo
        prompt = vp.expand(x.shape[0], -1, -1) if vp.numel() > 0 else None
        return encoder(x.type(self.dtype), prompt=prompt)

    def encode_text_prompted(self, modality):
        """CoOp-style prompted text encoding.

        Splice [SOS, learnable_ctx_modality, class_suffix] for every class, then run the
        (mostly frozen — LN trainable) text transformer and pick the [EOT] position.
        """
        if modality == "sketch":
            ctx = self.text_prompt_sketch
        else:
            ctx = self.text_prompt_photo

        n_cls = self.token_prefix.shape[0]
        if ctx.numel() == 0:
            # Degenerate fallback — no learnable tokens; replay the frozen full embedding.
            return self.clip.encode_text(self.tokenized_text)

        ctx_expanded = ctx.unsqueeze(0).expand(n_cls, -1, -1)                # (n_cls, n_ctx, d_t)
        x = torch.cat([self.token_prefix, ctx_expanded, self.token_suffix], dim=1)  # (n_cls, 77, d_t)

        x = x + self.clip.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)            # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)            # LND -> NLD
        x = self.clip.ln_final(x).type(self.dtype)

        # CLIP picks the embedding at the [EOT] position, found via argmax over token ids.
        eot_idx = self.tokenized_text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_idx] @ self.clip.text_projection
        return x

    def forward(self, x, classnames):
        sk_tensor = x[0]
        photo_tensor = x[1]
        neg_tensor = x[2]
        label = x[5] if len(x) >= 6 else x[3]

        sketch_feat = self.encode_visual(sk_tensor, "sketch")
        photo_feat = self.encode_visual(photo_tensor, "photo")
        neg_feat = self.encode_visual(neg_tensor, "photo")

        text_feat_photo = self.encode_text_prompted("photo")
        text_feat_sketch = self.encode_text_prompted("sketch")

        # L2-normalise for cosine similarity / cosine-distance triplet
        sketch_feat = sketch_feat / sketch_feat.norm(dim=-1, keepdim=True)
        photo_feat = photo_feat / photo_feat.norm(dim=-1, keepdim=True)
        neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
        text_feat_photo = text_feat_photo / text_feat_photo.norm(dim=-1, keepdim=True)
        text_feat_sketch = text_feat_sketch / text_feat_sketch.norm(dim=-1, keepdim=True)

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
        pass

    def on_fit_start(self):
        tokens_visual_sketch = self.model.visual_prompt_sketch.shape[0]
        tokens_visual_photo = self.model.visual_prompt_photo.shape[0]
        tokens_text_sketch = self.model.text_prompt_sketch.shape[0]
        tokens_text_photo = self.model.text_prompt_photo.shape[0]
        self.print(
            f"Learnable prompt tokens - visual/sketch: {tokens_visual_sketch}, "
            f"visual/photo: {tokens_visual_photo}, "
            f"text/sketch: {tokens_text_sketch}, text/photo: {tokens_text_photo}"
        )
        try:
            self.log('tokens_visual_sketch', tokens_visual_sketch, prog_bar=True, logger=True)
            self.log('tokens_visual_photo', tokens_visual_photo, prog_bar=True, logger=True)
            self.log('tokens_text_sketch', tokens_text_sketch, prog_bar=True, logger=True)
            self.log('tokens_text_photo', tokens_text_photo, prog_bar=True, logger=True)
        except Exception:
            pass

    def configure_optimizers(self):
        prompt_params = [
            p for p in [
                self.model.visual_prompt_sketch, self.model.visual_prompt_photo,
                self.model.text_prompt_sketch, self.model.text_prompt_photo,
            ]
            if p.requires_grad
        ]

        ln_params = []
        seen_ids = set(id(p) for p in prompt_params)

        # Visual branch LayerNorms (sketch + photo encoders)
        for branch in [self.model.visual_sketch, self.model.visual_photo]:
            for m in branch.modules():
                if isinstance(m, torch.nn.LayerNorm):
                    for p in m.parameters(recurse=False):
                        if p.requires_grad and id(p) not in seen_ids:
                            seen_ids.add(id(p))
                            ln_params.append(p)

        # Text encoder LayerNorms (inside self.clip.transformer + ln_final)
        for m in self.model.clip.transformer.modules():
            if isinstance(m, torch.nn.LayerNorm):
                for p in m.parameters(recurse=False):
                    if p.requires_grad and id(p) not in seen_ids:
                        seen_ids.add(id(p))
                        ln_params.append(p)
        for p in self.model.clip.ln_final.parameters(recurse=False):
            if p.requires_grad and id(p) not in seen_ids:
                seen_ids.add(id(p))
                ln_params.append(p)

        self.print(f"Trainable prompt params (visual + text, both modalities): {sum(p.numel() for p in prompt_params):,}")
        self.print(f"Trainable LayerNorm params (visual + text encoders): {sum(p.numel() for p in ln_params):,}")

        prompt_lr = getattr(self.cfg, 'prompt_lr', 1e-5)
        clip_ln_lr = getattr(self.cfg, 'clip_LN_lr', 1e-5)
        weight_decay = getattr(self.cfg, 'weight_decay', 1e-4)

        param_groups = [{'params': prompt_params, 'lr': prompt_lr}]
        if ln_params:
            param_groups.append({'params': ln_params, 'lr': clip_ln_lr})

        return torch.optim.Adam(param_groups, weight_decay=weight_decay)

    def training_step(self, batch, batch_idx):
        from src.losses_hicropl import loss_fn_hicropl
        features = self.model(batch, self.classnames)
        loss = loss_fn_hicropl(self.args, features)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)

        return loss

    def extract_eval_features(self, tensor, modality):
        feat = self.model.encode_visual(tensor, modality)
        return feat / feat.norm(dim=-1, keepdim=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._validation_step_category(batch, batch_idx, dataloader_idx)

    def _validation_step_category(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 3:
            tensor, label, type_data = batch
        else:
            tensor, label = batch

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
        query_features = torch.cat(self.test_sketch_features, dim=0).to(self.device)

        all_photo_category = torch.cat(self.test_photo_labels, dim=0).to(self.device)
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
