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
      - SINGLE shared CLIP backbone (`self.clip`), not two separate encoders.
      - Trainable visual params come from `clip.apply(freeze_all_but_bn)` which leaves
        LN + MHA QKV projections + naked Parameters open (~21.7M visual params).
      - Two visual prompts (sketch / photo), shallow, injected at the first ViT layer.

    Extensions we keep on top of the official baseline (not in `model_LN_prompt.py`):
      - Two CoOp-style learnable text prompts (sketch / photo), shallow.
      - Text-template classification loss (L_class) using these prompted text features.
    """

    def __init__(self, cfg, clip_model, clip_model_frozen=None, classnames=None):
        super().__init__()
        self.cfg = cfg

        if classnames is None or len(classnames) == 0:
            raise ValueError("CustomCLIP requires non-empty classnames during initialization.")

        original_device = next(clip_model.parameters()).device
        self.dtype = clip_model.dtype

        # Single shared CLIP backbone (official CLIP-AT uses `self.clip` for both modalities).
        self.clip = copy.deepcopy(clip_model).to(original_device)
        self.clip.apply(freeze_all_but_bn)

        # Optionally freeze the whole text branch (incl. its LayerNorm) so only the visual
        # branch (LN + prompts) learns; text features become fixed zero-shot CLIP features.
        if getattr(cfg, "freeze_text", False):
            freeze_model(self.clip.transformer)        # text transformer (visual is self.clip.visual)
            freeze_model(self.clip.token_embedding)
            freeze_model(self.clip.ln_final)
            self.clip.positional_embedding.requires_grad_(False)   # text positional embedding
            self.clip.text_projection.requires_grad_(False)
            print("freeze_text=True: text branch fully frozen (LN included); logit_scale stays shared.")

        def _count_trainable(m):
            total = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable

        c_tot, c_tr = _count_trainable(self.clip)
        print(f"clip (visual + text, freeze_all_but_bn): trainable {c_tr:,} / total {c_tot:,}")

        self.logit_scale = self.clip.logit_scale

        # Per-modality hard templates. The first n_ctx context tokens after [SOS] will be
        # REPLACED at forward time by learnable per-modality prompts (CoOp style); the literal
        # context words here only set positional structure. The class-name suffix differs
        # only in whether tokens 1..1+n_ctx originally read 'a photo of a' or 'a sketch of a'.
        ctx_init_photo = getattr(cfg, "ctx_init", "a photo of a")
        ctx_init_sketch = getattr(cfg, "ctx_init_sketch", "a sketch of a")
        prompts_photo = [f"{ctx_init_photo} {name}.".replace("_", " ").strip() for name in classnames]
        prompts_sketch = [f"{ctx_init_sketch} {name}.".replace("_", " ").strip() for name in classnames]
        self.register_buffer("tokenized_photo", _clip.tokenize(prompts_photo).to(original_device))
        self.register_buffer("tokenized_sketch", _clip.tokenize(prompts_sketch).to(original_device))

        # Two separate shallow visual prompts (official CLIP-AT: sk_prompt, img_prompt).
        prompt_dim_v = self.clip.visual.conv1.weight.shape[0]
        n_ctx = int(getattr(cfg, "n_ctx", 3))
        self.n_ctx = n_ctx
        self.visual_prompt_sketch = nn.Parameter(torch.empty(n_ctx, prompt_dim_v, dtype=self.dtype))
        self.visual_prompt_photo = nn.Parameter(torch.empty(n_ctx, prompt_dim_v, dtype=self.dtype))
        if n_ctx > 0:
            nn.init.normal_(self.visual_prompt_sketch, std=0.02)
            nn.init.normal_(self.visual_prompt_photo, std=0.02)

        # Text branch mode:
        #   'template'  -> CLIP-AT style: hard frozen template ("a photo/sketch of a {cls}"),
        #                  encoded by the (LN-trainable) text encoder, NO learnable text prompt.
        #   'learnable' -> CoOp-style: learnable context tokens replace the template words.
        self.text_prompt_mode = getattr(cfg, "text_prompt_mode", "template")

        # Per-modality context tokens. Kept as Parameters for both modes, but frozen (unused)
        # under 'template' so the optimizer skips them.
        prompt_dim_t = self.clip.ln_final.weight.shape[0]
        self.text_prompt_sketch = nn.Parameter(torch.empty(n_ctx, prompt_dim_t, dtype=self.dtype))
        self.text_prompt_photo = nn.Parameter(torch.empty(n_ctx, prompt_dim_t, dtype=self.dtype))
        if n_ctx > 0:
            nn.init.normal_(self.text_prompt_sketch, std=0.02)
            nn.init.normal_(self.text_prompt_photo, std=0.02)
        if self.text_prompt_mode == "template":
            self.text_prompt_sketch.requires_grad_(False)
            self.text_prompt_photo.requires_grad_(False)

        # Cache the frozen [SOS] prefix and [class + EOT + pad] suffix embeddings PER MODALITY.
        with torch.no_grad():
            embed_photo = self.clip.token_embedding(self.tokenized_photo).type(self.dtype)
            embed_sketch = self.clip.token_embedding(self.tokenized_sketch).type(self.dtype)
        # full_embed: (n_cls, 77, d_t). prefix = token 0 ([SOS]); suffix = tokens 1+n_ctx :
        self.register_buffer("token_prefix_photo", embed_photo[:, :1, :])
        self.register_buffer("token_suffix_photo", embed_photo[:, 1 + n_ctx:, :])
        self.register_buffer("token_prefix_sketch", embed_sketch[:, :1, :])
        self.register_buffer("token_suffix_sketch", embed_sketch[:, 1 + n_ctx:, :])

    def encode_visual(self, x, modality):
        """Encode image through the shared CLIP visual encoder with the modality-specific prompt."""
        vp = self.visual_prompt_sketch if modality == "sketch" else self.visual_prompt_photo
        prompt = vp.expand(x.shape[0], -1, -1) if vp.numel() > 0 else None
        return self.clip.encode_image(x.type(self.dtype), prompt=prompt)

    def encode_text_prompted(self, modality):
        """CoOp-style prompted text encoding (per modality).

        Splice [SOS, learnable_ctx_modality, class_suffix_modality] for every class, then run
        the (LN-trainable) text transformer and pick the [EOT] position.
        """
        if modality == "sketch":
            ctx = self.text_prompt_sketch
            prefix = self.token_prefix_sketch
            suffix = self.token_suffix_sketch
            tokenized = self.tokenized_sketch
        else:
            ctx = self.text_prompt_photo
            prefix = self.token_prefix_photo
            suffix = self.token_suffix_photo
            tokenized = self.tokenized_photo

        n_cls = prefix.shape[0]
        # CLIP-AT template mode: encode the hard template directly (no learnable context).
        if self.text_prompt_mode == "template" or ctx.numel() == 0:
            return self.clip.encode_text(tokenized)

        ctx_expanded = ctx.unsqueeze(0).expand(n_cls, -1, -1)                  # (n_cls, n_ctx, d_t)
        x = torch.cat([prefix, ctx_expanded, suffix], dim=1)                   # (n_cls, 77, d_t)

        x = x + self.clip.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)            # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)            # LND -> NLD
        x = self.clip.ln_final(x).type(self.dtype)

        eot_idx = tokenized.argmax(dim=-1)
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
        """Official CLIP-AT optimizer pattern (Sain et al. CVPR'23).

        Two groups:
          1. ALL `self.clip.parameters()` at `clip_LN_lr` — Adam only updates the ones with
             `requires_grad=True`, which after `freeze_all_but_bn` includes LN + naked
             Parameters + MHA QKV projections (~21.7M visual + ~22M text).
          2. Learnable prompts (visual + text, both modalities) at `prompt_lr`.
        """
        prompt_params = [
            p for p in [
                self.model.visual_prompt_sketch, self.model.visual_prompt_photo,
                self.model.text_prompt_sketch, self.model.text_prompt_photo,
            ]
            if p.requires_grad
        ]

        clip_params = list(self.model.clip.parameters())
        clip_trainable = sum(p.numel() for p in clip_params if p.requires_grad)

        self.print(f"Trainable prompt params (visual + text, both modalities): {sum(p.numel() for p in prompt_params):,}")
        self.print(f"Trainable clip params (LN + naked + MHA QKV, official CLIP-AT pattern): {clip_trainable:,}")

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
