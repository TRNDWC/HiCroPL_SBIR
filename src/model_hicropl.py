import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

from src.clip import clip as _clip


def freeze_model(m):
    """Freeze all parameters of the given module."""
    for param in m.parameters():
        param.requires_grad_(False)


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


def freeze_all_but_bn(m):
    """Official CLIP-AT freeze hook (Sain et al. CVPR'23).

    Iterates submodules and freezes `.weight`/`.bias` unless the module is a LayerNorm.
    Side-effect by design: it does NOT touch parameters that aren't exposed as `.weight`/`.bias`,
    so the following stay trainable in the visual & text encoders (matches official repo):
      - `nn.MultiheadAttention.in_proj_weight`, `in_proj_bias`
      - Naked Parameters on the module itself: `class_embedding`, `positional_embedding`,
        `proj`, `token_embedding`, `text_projection`
    """
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad_(False)


class CustomCLIP(nn.Module):
    """CLIP-AT-aligned model (Sain et al. CVPR'23).

    Faithful to the OFFICIAL repo (aneeshan95/Sketch_LVM, file `model_LN_prompt.py`):
      - TWO separate CLIP backbones (`self.clip_photo`, `self.clip_sketch`), one per modality.
      - Trainable visual params come from `clip.apply(freeze_all_but_bn)` which leaves
        LN + MHA QKV projections + naked Parameters open (~21.7M visual params).
      - Two visual prompts (sketch / photo), shallow, injected at the first ViT layer.

    Freeze pattern (`clip.apply(freeze_all_but_bn)`):
      - LN weight/bias: trainable (mọi block visual + text)
      - MHA in_proj_weight / in_proj_bias: trainable (naked params, không bị freeze_all_but_bn chạm vào)
      - class_embedding, positional_embedding, proj, token_embedding, text_projection: trainable
      - Conv/Linear weight/bias: frozen
    """

    def __init__(self, cfg, clip_model, clip_model_frozen=None, classnames=None):
        super().__init__()
        self.cfg = cfg

        if classnames is None or len(classnames) == 0:
            raise ValueError("CustomCLIP requires non-empty classnames during initialization.")

        original_device = next(clip_model.parameters()).device
        self.dtype = clip_model.dtype

        # 2 visual CLIP backbones — one per modality (photo / sketch).
        self.clip_photo = copy.deepcopy(clip_model).to(original_device)
        self.clip_sketch = copy.deepcopy(clip_model).to(original_device)
        self.clip_photo.apply(freeze_all_but_bn)
        self.clip_sketch.apply(freeze_all_but_bn)

        # [Thay đổi 1] model_distill: shared text encoder + aug feature encoder.
        # Dùng clip_model_frozen (CoOp/IVLP, vision_depth=0) — apply freeze_all_but_bn
        # giống CoPrompt (KHÔNG freeze hoàn toàn, LN vẫn trainable).
        if clip_model_frozen is None:
            clip_model_frozen = clip_model
        self.model_distill = copy.deepcopy(clip_model_frozen).to(original_device)
        self.model_distill.apply(freeze_all_but_bn)

        def _count_trainable(m):
            total = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable

        for name, m in [("clip_photo", self.clip_photo),
                         ("clip_sketch", self.clip_sketch),
                         ("model_distill", self.model_distill)]:
            tot, tr = _count_trainable(m)
            print(f"{name}: trainable {tr:,} / total {tot:,}")

        self.logit_scale_photo = self.clip_photo.logit_scale
        self.logit_scale_sketch = self.clip_sketch.logit_scale

        ctx_init_photo = getattr(cfg, "ctx_init", "a photo of a")
        ctx_init_sketch = getattr(cfg, "ctx_init_sketch", "a sketch of a")

        # Two separate shallow visual prompts.
        prompt_dim_v = self.clip_photo.visual.conv1.weight.shape[0]
        n_ctx = int(getattr(cfg, "n_ctx", 3))
        self.n_ctx = n_ctx
        self.visual_prompt_sketch = nn.Parameter(torch.empty(n_ctx, prompt_dim_v, dtype=self.dtype))
        self.visual_prompt_photo = nn.Parameter(torch.empty(n_ctx, prompt_dim_v, dtype=self.dtype))
        if n_ctx > 0:
            nn.init.normal_(self.visual_prompt_sketch, std=0.02)
            nn.init.normal_(self.visual_prompt_photo, std=0.02)

        # [Thay đổi 2] Learnable text tokens — khởi tạo từ model_distill.token_embedding
        # (shared cho cả 2 nhánh, giống CoPrompt).
        with torch.no_grad():
            def _ctx_from_init(text, n):
                tok = _clip.tokenize(text).to(original_device)
                emb = self.model_distill.token_embedding(tok).type(self.dtype)
                return emb[0, 1:1 + n, :].clone()
            ctx_photo_init = _ctx_from_init(ctx_init_photo, n_ctx)
            ctx_sketch_init = _ctx_from_init(ctx_init_sketch, n_ctx)
        self.text_ctx_photo = nn.Parameter(ctx_photo_init)
        self.text_ctx_sketch = nn.Parameter(ctx_sketch_init)

        # [Thay đổi 2] Prefix/suffix buffers dùng model_distill.token_embedding (shared).
        placeholder = " ".join(["X"] * n_ctx)
        def _build_prefix_suffix(templates):
            tok = _clip.tokenize(templates).to(original_device)
            with torch.no_grad():
                emb = self.model_distill.token_embedding(tok).type(self.dtype)
            return emb[:, :1, :], emb[:, 1 + n_ctx:, :], tok

        ph_templates = [f"{placeholder} {n}.".replace("_", " ") for n in classnames]
        sk_templates = [f"{placeholder} {n}.".replace("_", " ") for n in classnames]
        ph_prefix, ph_suffix, ph_tok = _build_prefix_suffix(ph_templates)
        sk_prefix, sk_suffix, sk_tok = _build_prefix_suffix(sk_templates)
        self.register_buffer("text_prefix_photo", ph_prefix)
        self.register_buffer("text_suffix_photo", ph_suffix)
        self.register_buffer("text_prefix_sketch", sk_prefix)
        self.register_buffer("text_suffix_sketch", sk_suffix)
        self.register_buffer("text_eot_photo", ph_tok)
        self.register_buffer("text_eot_sketch", sk_tok)

    def encode_visual(self, x, modality):
        clip = self.clip_photo if modality == "photo" else self.clip_sketch
        vp = self.visual_prompt_photo if modality == "photo" else self.visual_prompt_sketch
        prompt = vp.expand(x.shape[0], -1, -1) if vp.numel() > 0 else None
        return clip.encode_image(x.type(self.dtype), prompt=prompt)

    def encode_text_with_prompt(self, modality):
        """Encode text with learnable shallow context tokens.

        [Thay đổi 2] Dùng model_distill (shared text encoder) cho cả 2 nhánh,
        chỉ ctx (learnable tokens) là khác nhau giữa photo và sketch.
        """
        if modality == "photo":
            ctx, prefix, suffix, eot = (
                self.text_ctx_photo, self.text_prefix_photo,
                self.text_suffix_photo, self.text_eot_photo,
            )
        else:
            ctx, prefix, suffix, eot = (
                self.text_ctx_sketch, self.text_prefix_sketch,
                self.text_suffix_sketch, self.text_eot_sketch,
            )
        n_cls = prefix.shape[0]
        ctx_exp = ctx.unsqueeze(0).expand(n_cls, -1, -1)
        prompts = torch.cat([prefix, ctx_exp, suffix], dim=1).type(self.dtype)

        x = prompts + self.model_distill.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.model_distill.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.model_distill.ln_final(x).type(self.dtype)
        x = x[torch.arange(n_cls), eot.argmax(dim=-1)] @ self.model_distill.text_projection
        return x

    def forward(self, x, classnames):
        sk_tensor = x[0]
        photo_tensor = x[1]
        neg_tensor = x[2]
        label = x[5] if len(x) >= 6 else x[3]

        sketch_feat = self.encode_visual(sk_tensor, "sketch")
        photo_feat = self.encode_visual(photo_tensor, "photo")
        neg_feat = self.encode_visual(neg_tensor, "photo")

        text_feat_photo = self.encode_text_with_prompt("photo")
        text_feat_sketch = self.encode_text_with_prompt("sketch")

        sketch_feat = sketch_feat / sketch_feat.norm(dim=-1, keepdim=True)
        photo_feat = photo_feat / photo_feat.norm(dim=-1, keepdim=True)
        neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
        text_feat_photo = text_feat_photo / text_feat_photo.norm(dim=-1, keepdim=True)
        text_feat_sketch = text_feat_sketch / text_feat_sketch.norm(dim=-1, keepdim=True)

        logits_photo = self.logit_scale_photo.exp() * photo_feat @ text_feat_photo.t()
        logits_sketch = self.logit_scale_sketch.exp() * sketch_feat @ text_feat_sketch.t()

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
        tokens_text_photo = self.model.text_ctx_photo.shape[0]
        tokens_text_sketch = self.model.text_ctx_sketch.shape[0]
        self.print(
            f"Learnable tokens - visual/sketch: {tokens_visual_sketch}, "
            f"visual/photo: {tokens_visual_photo}, "
            f"text/photo: {tokens_text_photo}, text/sketch: {tokens_text_sketch}"
        )
        try:
            self.log('tokens_visual_sketch', tokens_visual_sketch, prog_bar=True, logger=True)
            self.log('tokens_visual_photo', tokens_visual_photo, prog_bar=True, logger=True)
            self.log('tokens_text_photo', tokens_text_photo, prog_bar=True, logger=True)
            self.log('tokens_text_sketch', tokens_text_sketch, prog_bar=True, logger=True)
        except Exception:
            pass

    def configure_optimizers(self):
        """Official CLIP-AT optimizer pattern (Sain et al. CVPR'23).

        Two groups:
          1. ALL `self.clip_photo` + `self.clip_sketch` parameters at `clip_LN_lr` — Adam only updates the ones with
             `requires_grad=True`, which after `freeze_all_but_bn` includes LN + naked
             Parameters + MHA QKV projections (~21.7M visual + ~22M text).
          2. Learnable prompts (visual + text, both modalities) at `prompt_lr`.
        """
        prompt_params = [
            p for p in [
                self.model.visual_prompt_sketch, self.model.visual_prompt_photo,
                self.model.text_ctx_photo, self.model.text_ctx_sketch,
            ]
            if p.requires_grad
        ]

        clip_params = (list(self.model.clip_photo.parameters()) +
                       list(self.model.clip_sketch.parameters()) +
                       list(self.model.model_distill.parameters()))
        clip_trainable = sum(p.numel() for p in clip_params if p.requires_grad)

        self.print(f"Trainable prompt params (visual + text, both modalities): {sum(p.numel() for p in prompt_params):,}")
        self.print(f"Trainable clip params (clip_photo + clip_sketch + model_distill): {clip_trainable:,}")

        prompt_lr = getattr(self.cfg, 'prompt_lr', 1e-5)
        clip_ln_lr = getattr(self.cfg, 'clip_LN_lr', 1e-5)
        weight_decay = getattr(self.cfg, 'weight_decay', 0.0)

        return torch.optim.Adam([
            {'params': clip_params, 'lr': clip_ln_lr},
            {'params': prompt_params, 'lr': prompt_lr},
        ], weight_decay=weight_decay)

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
        if dataset in ("sketchy_2", "sketchy_ext"):
            map_k = 200      # mAP@200 lenient (khớp CDUF / doodle2search)
            p_k = 200
        elif dataset == "quickdraw":
            map_k = 0        # mAP@all
            p_k = 200
        else:
            map_k = 0        # mAP@all
            p_k = 100

        n_g = gallery_features.shape[0]
        ranks = torch.arange(1, n_g + 1, device=self.device, dtype=torch.float32)
        kk = min(map_k, n_g) if map_k != 0 else n_g

        ap_lenient = torch.zeros(len(query_features), device=self.device)  # mAP@k ÷ relevant-in-top-k (CDUF/doodle2search)
        ap_all = torch.zeros(len(query_features), device=self.device)      # mAP@all ÷ R (CLIP-AT / ZSE-SBIR map_all)
        ap_strict = torch.zeros(len(query_features), device=self.device)   # mAP@k ÷ min(R,k) (SAKE / ZSE-SBIR strict)
        precision = torch.zeros(len(query_features), device=self.device)

        for idx in range(len(query_features)):
            category = all_sketch_category[idx]
            sim = similarity_matrix[idx]
            target = (all_photo_category == category)
            R = target.sum().clamp(min=1).float()

            # Xếp gallery theo similarity giảm dần, lấy relevance theo thứ hạng.
            order = torch.argsort(sim, descending=True)
            rel = target[order].float()
            prec_at = torch.cumsum(rel, dim=0) / ranks           # precision@mỗi rank

            # Tử số chung cho top-k = tổng precision@hit của các relevant trong top-k.
            hit_k = (prec_at[:kk] * rel[:kk]).sum()
            ap_lenient[idx] = hit_k / rel[:kk].sum().clamp(min=1)                                  # ÷ rel-in-top-k
            ap_strict[idx] = hit_k / torch.minimum(R, torch.tensor(float(kk), device=self.device))  # ÷ min(R,k)
            ap_all[idx] = (prec_at * rel).sum() / R                                                # ÷ R, full ranking

            # P@p_k = (relevant trong top-p_k) / p_k
            precision[idx] = rel[:min(p_k, n_g)].sum() / p_k

        m_lenient = torch.mean(ap_lenient)
        m_all = torch.mean(ap_all)
        m_strict = torch.mean(ap_strict)
        mean_precision = torch.mean(precision)

        # Monitor: ext -> mAP@200 lenient (val_map_200); còn lại -> mAP@all (lenient@all == mAP@all).
        mAP = m_lenient
        self.log("mAP", mAP, on_step=False, on_epoch=True)
        self.log("val_mAP", mAP, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_map_all", m_all, on_step=False, on_epoch=True)
        if map_k != 0:
            self.log(f"val_map_{map_k}", m_lenient, on_step=False, on_epoch=True)
            self.log(f"val_map_{map_k}_strict", m_strict, on_step=False, on_epoch=True)
        self.log(f"P@{p_k}", mean_precision, on_step=False, on_epoch=True)
        self.log(f"val_P@{p_k}", mean_precision, on_step=False, on_epoch=True)
        self.log(f"val_p_{p_k}", mean_precision, on_step=False, on_epoch=True)
        self.log("best_mAP", self.best_metric, on_step=False, on_epoch=True, prog_bar=False)

        if self.global_step > 0:
            self.best_metric = max(self.best_metric, mAP.item())

        if map_k != 0:
            self.print(
                'mAP@{} lenient: {:.4f} | mAP@all: {:.4f} | mAP@{} strict: {:.4f} | '
                'P@{}: {:.4f} | Best mAP: {:.4f}'.format(
                    map_k, m_lenient.item(), m_all.item(), map_k, m_strict.item(),
                    p_k, mean_precision.item(), self.best_metric))
        else:
            self.print('mAP@all: {:.4f} | P@{}: {:.4f} | Best mAP: {:.4f}'.format(
                m_all.item(), p_k, mean_precision.item(), self.best_metric))

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
