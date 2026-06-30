import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision

from src.clip import clip as _clip


def freeze_model(m):
    for param in m.parameters():
        param.requires_grad_(False)


def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad_(False)


class CustomCLIP(nn.Module):
    """
    Simplified SBIR wrapper with 2 CLIP instances (like CoPrompt):
      - clip_model       : student, provides logit_scale
      - clip_model_frozen: teacher, provides ph_encoder, sk_encoder, text components

    No visual prompts. Text prompts via learnable ctx_photo / ctx_sketch (CoOp-style).
    """

    def __init__(self, cfg, clip_model, clip_model_frozen, classnames=None):
        super().__init__()
        self.cfg = cfg

        if not classnames:
            raise ValueError("CustomCLIP requires non-empty classnames.")

        original_device = next(clip_model.parameters()).device
        self.dtype = clip_model.dtype

        # 1. Freeze both CLIP models; only LayerNorm weights remain trainable
        clip_model.apply(freeze_all_but_bn)
        clip_model_frozen.apply(freeze_all_but_bn)

        # 2. Visual encoders — plain VisionTransformer (forward(x), no prompt injection)
        #    Deepcopy gives each branch independent LN weights
        self.ph_encoder = copy.deepcopy(clip_model_frozen.visual).to(original_device)
        self.sk_encoder = copy.deepcopy(clip_model_frozen.visual).to(original_device)

        # 2b. Frozen reference encoder for consistency loss target (CoPrompt-style distillation)
        #     All params frozen — provides stable anchor for aug features (no LN update here)
        self.teacher_encoder = copy.deepcopy(clip_model_frozen.visual).to(original_device)
        freeze_model(self.teacher_encoder)

        # 3. Logit scale from student
        self.logit_scale = clip_model.logit_scale

        # 4. Text encoder components from teacher (shared reference, LN trainable)
        self.text_transformer = clip_model_frozen.transformer
        self.text_ln_final    = clip_model_frozen.ln_final
        self.register_buffer("text_pos_embed", clip_model_frozen.positional_embedding.data.clone())
        self.register_buffer("text_proj",      clip_model_frozen.text_projection.data.clone())

        # 5. Learnable text context vectors
        n_ctx         = int(getattr(cfg, 'n_ctx', 4))
        prefix_photo  = getattr(cfg, 'ctx_init',        'a photo of a')
        prefix_sketch = getattr(cfg, 'ctx_init_sketch', 'a sketch of a')

        with torch.no_grad():
            ph_prefix_tok = _clip.tokenize([prefix_photo]).to(original_device)
            sk_prefix_tok = _clip.tokenize([prefix_sketch]).to(original_device)
            ph_prefix_emb = clip_model_frozen.token_embedding(ph_prefix_tok).type(self.dtype)
            sk_prefix_emb = clip_model_frozen.token_embedding(sk_prefix_tok).type(self.dtype)
            ctx_photo_init  = ph_prefix_emb[0, 1:1 + n_ctx, :].clone()
            ctx_sketch_init = sk_prefix_emb[0, 1:1 + n_ctx, :].clone()

        self.ctx_photo  = nn.Parameter(ctx_photo_init)
        self.ctx_sketch = nn.Parameter(ctx_sketch_init)
        self.dropout_ctx = nn.Dropout(p=0.1)

        # 6. Build prefix/suffix buffers for all classes
        eot_id = int(_clip.tokenize([""])[0, 1].item())
        ph_prefix_len = int((ph_prefix_tok[0] == eot_id).nonzero()[0].item())
        sk_prefix_len = int((sk_prefix_tok[0] == eot_id).nonzero()[0].item())

        placeholder = " ".join(["X"] * n_ctx)
        ph_tmpl = [f"{prefix_photo} {placeholder} {n}.".replace("_", " ") for n in classnames]
        sk_tmpl = [f"{prefix_sketch} {placeholder} {n}.".replace("_", " ") for n in classnames]

        with torch.no_grad():
            ph_cls_tok = _clip.tokenize(ph_tmpl).to(original_device)
            sk_cls_tok = _clip.tokenize(sk_tmpl).to(original_device)
            ph_cls_emb = clip_model_frozen.token_embedding(ph_cls_tok).type(self.dtype)
            sk_cls_emb = clip_model_frozen.token_embedding(sk_cls_tok).type(self.dtype)

        self.register_buffer("text_prefix_photo",  ph_cls_emb[:, :ph_prefix_len, :])
        self.register_buffer("text_suffix_photo",  ph_cls_emb[:, ph_prefix_len + n_ctx:, :])
        self.register_buffer("text_prefix_sketch", sk_cls_emb[:, :sk_prefix_len, :])
        self.register_buffer("text_suffix_sketch", sk_cls_emb[:, sk_prefix_len + n_ctx:, :])
        self.register_buffer("text_tok_photo",  ph_cls_tok)
        self.register_buffer("text_tok_sketch", sk_cls_tok)

        print(f"Text ctx: n_ctx={n_ctx}, ph_prefix_len={ph_prefix_len}, sk_prefix_len={sk_prefix_len}")

        def _count_trainable(m):
            total = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable

        for name, module in (("ph_encoder", self.ph_encoder), ("sk_encoder", self.sk_encoder)):
            tot, tr = _count_trainable(module)
            print(f"{name}: trainable {tr:,} / total {tot:,} params")

    def encode_text(self, modality):
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
        x = x.permute(1, 0, 2)             # NLD -> LND
        x = self.text_transformer(x)       # IVLP transformer: plain tensor in/out
        x = x.permute(1, 0, 2)             # LND -> NLD
        x = self.text_ln_final(x).type(self.dtype)
        x = x[torch.arange(n_cls), tok.argmax(dim=-1)] @ self.text_proj
        return x / x.norm(dim=-1, keepdim=True)

    def forward(self, x, classnames):
        sk_tensor, photo_tensor, neg_tensor, sk_aug_tensor, photo_aug_tensor, label = x[:6]

        # Visual encoding (no visual prompts)
        photo_raw  = self.ph_encoder(photo_tensor.type(self.dtype))
        sketch_raw = self.sk_encoder(sk_tensor.type(self.dtype))
        neg_raw    = self.ph_encoder(neg_tensor.type(self.dtype))

        photo_feat  = photo_raw  / photo_raw.norm(dim=-1, keepdim=True)
        sketch_feat = sketch_raw / sketch_raw.norm(dim=-1, keepdim=True)
        neg_feat    = neg_raw    / neg_raw.norm(dim=-1, keepdim=True)

        # Augmented features for L2 consistency — frozen teacher as stable anchor
        # (CoPrompt-style: student original vs teacher aug, not student vs student)
        with torch.no_grad():
            photo_aug_raw  = self.teacher_encoder(photo_aug_tensor.type(self.dtype))
            sketch_aug_raw = self.teacher_encoder(sk_aug_tensor.type(self.dtype))
        photo_aug_feat  = photo_aug_raw  / photo_aug_raw.norm(dim=-1, keepdim=True)
        sketch_aug_feat = sketch_aug_raw / sketch_aug_raw.norm(dim=-1, keepdim=True)

        # Text encoding
        text_feat_photo  = self.encode_text("photo")
        text_feat_sketch = self.encode_text("sketch")

        # Logits
        logit_scale   = self.logit_scale.exp()
        logits_photo  = logit_scale * photo_feat  @ text_feat_photo.t()
        logits_sketch = logit_scale * sketch_feat @ text_feat_sketch.t()

        return (
            photo_feat, logits_photo,
            sketch_feat, logits_sketch,
            neg_feat, label,
            photo_aug_feat, sketch_aug_feat,
            None, None,   # logits_photo_aug, logits_sketch_aug (không dùng)
            text_feat_photo, text_feat_sketch,
            None, None,   # text_distill_photo, text_distill_sketch (L3 removed)
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

    def on_fit_start(self):
        try:
            tokens_text_photo  = self.model.ctx_photo.shape[0]
            tokens_text_sketch = self.model.ctx_sketch.shape[0]
        except Exception:
            tokens_text_photo = tokens_text_sketch = 0

        self.print(
            f"Learnable tokens — text/photo: {tokens_text_photo}, "
            f"text/sketch: {tokens_text_sketch}"
        )
        try:
            self.log('tokens_text_photo',  tokens_text_photo,  prog_bar=True, logger=True)
            self.log('tokens_text_sketch', tokens_text_sketch, prog_bar=True, logger=True)
        except Exception:
            pass

    def configure_optimizers(self):
        def add_unique(candidates, out_list, seen):
            for p in candidates:
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    out_list.append(p)

        seen = set()
        prompt_params = []
        add_unique([self.model.ctx_photo, self.model.ctx_sketch], prompt_params, seen)

        ln_params = []
        # LN from ph/sk encoders and text transformer components
        for module in [self.model.ph_encoder, self.model.sk_encoder,
                       self.model.text_transformer, self.model.text_ln_final]:
            add_unique(module.parameters(), ln_params, seen)

        extra = []
        for _, p in self.model.named_parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                extra.append(p)

        non_prompt_params = ln_params + extra

        self.print(f"Trainable prompt params:     {sum(p.numel() for p in prompt_params):,}")
        self.print(f"Trainable non-prompt params: {sum(p.numel() for p in non_prompt_params):,}")

        prompt_lr    = getattr(self.cfg, 'prompt_lr',    1e-5)
        clip_ln_lr   = getattr(self.cfg, 'clip_LN_lr',   1e-5)
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
        self.log('loss',       loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        return loss

    def extract_eval_features(self, tensor, modality):
        encoder = self.model.ph_encoder if modality == 'photo' else self.model.sk_encoder
        feat = encoder(tensor.type(self.model.dtype))
        return feat / feat.norm(dim=-1, keepdim=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._validation_step_category(batch, batch_idx, dataloader_idx)

    def _validation_step_category(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 3:
            tensor, label, _ = batch
        else:
            tensor, label = batch

        if dataloader_idx == 0:
            feat = self.extract_eval_features(tensor, modality='sketch')
            self.test_sketch_features.append(feat.cpu().detach())
            self.test_sketch_labels.append(label.cpu().detach())
        elif dataloader_idx == 1:
            feat = self.extract_eval_features(tensor, modality='photo')
            self.test_photo_features.append(feat.cpu().detach())
            self.test_photo_labels.append(label.cpu().detach())

    def on_validation_epoch_end(self):
        return self._on_validation_epoch_end_category()

    def _on_validation_epoch_end_category(self):
        if not self.test_photo_features or not self.test_sketch_features:
            self.print("Warning: Missing features for validation. Skipping metrics.")
            return

        gallery_features = torch.cat(self.test_photo_features,  dim=0).to(self.device)
        query_features   = torch.cat(self.test_sketch_features, dim=0).to(self.device)
        all_photo_labels  = torch.cat(self.test_photo_labels,   dim=0).to(self.device)
        all_sketch_labels = torch.cat(self.test_sketch_labels,  dim=0).to(self.device)

        similarity_matrix = query_features @ gallery_features.t()

        dataset = getattr(self.args, 'dataset', 'sketchy')
        if dataset in ("sketchy_2", "sketchy_ext"):
            map_k, p_k = 200, 200
        elif dataset == "quickdraw":
            map_k, p_k = 0, 200
        else:
            map_k, p_k = 0, 100

        ap        = torch.zeros(len(query_features),  device=self.device)
        precision = torch.zeros(len(query_features),  device=self.device)

        for idx in range(len(query_features)):
            category = all_sketch_labels[idx]
            distance = similarity_matrix[idx]
            target   = (all_photo_labels == category)

            if map_k != 0:
                ap[idx] = retrieval_average_precision(distance, target, top_k=min(map_k, len(gallery_features)))
            else:
                ap[idx] = retrieval_average_precision(distance, target)

            precision[idx] = retrieval_precision(distance, target, top_k=p_k)

        mAP            = torch.mean(ap)
        mean_precision = torch.mean(precision)

        self.log("mAP",     mAP,            on_step=False, on_epoch=True)
        self.log(f"P@{p_k}", mean_precision, on_step=False, on_epoch=True)
        self.log("val_mAP",     mAP,            on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"val_P@{p_k}", mean_precision, on_step=False, on_epoch=True)
        self.log("best_mAP",    self.best_metric, on_step=False, on_epoch=True, prog_bar=False)

        if map_k != 0:
            self.log(f"val_map_{map_k}", mAP, on_step=False, on_epoch=True)
        else:
            self.log("val_map_all", mAP, on_step=False, on_epoch=True)
        self.log(f"val_p_{p_k}", mean_precision, on_step=False, on_epoch=True)

        if self.global_step > 0:
            self.best_metric = max(self.best_metric, mAP.item())

        if map_k != 0:
            self.print(f"mAP@{map_k}: {mAP:.4f}, P@{p_k}: {mean_precision:.4f}, Best mAP: {self.best_metric:.4f}")
        else:
            self.print(f"mAP@all: {mAP:.4f}, P@{p_k}: {mean_precision:.4f}, Best mAP: {self.best_metric:.4f}")

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
