import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.clip import clip as _clip


def freeze_all_but_bn(m):
    """Freeze .weight/.bias trên mọi module trừ LayerNorm.

    Sau khi apply, các tham số còn trainable:
      - LayerNorm weight/bias
      - MHA in_proj_weight, in_proj_bias  (naked params, không bị chạm)
      - Naked Parameters: class_embedding, positional_embedding, proj, text_projection
    """
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad_(False)


class CustomCLIP(nn.Module):
    """Baseline không có learnable prompt.

    Visual encoder (photo/sketch) tách biệt, freeze_all_but_bn —
    chỉ LayerNorm + MHA in_proj + naked params được train.
    Text features được tính một lần từ frozen CLIP với template cố định
    và lưu vào buffer — không có learnable token nào.
    """

    def __init__(self, cfg, clip_model, clip_model_frozen=None, classnames=None):
        super().__init__()
        self.cfg = cfg

        if classnames is None or len(classnames) == 0:
            raise ValueError("CustomCLIP requires non-empty classnames during initialization.")

        original_device = next(clip_model.parameters()).device
        self.dtype = clip_model.dtype

        clip_model.apply(freeze_all_but_bn)
        self.ph_encoder = copy.deepcopy(clip_model.visual).to(original_device)
        self.sk_encoder = copy.deepcopy(clip_model.visual).to(original_device)

        def _count_trainable(m):
            total = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable

        for name, m in [("ph_encoder", self.ph_encoder), ("sk_encoder", self.sk_encoder)]:
            tot, tr = _count_trainable(m)
            print(f"{name}: trainable {tr:,} / total {tot:,}")

        self.logit_scale = clip_model.logit_scale

        # ── Text encoder: đóng băng tất cả trừ LayerNorm ──
        text_src = clip_model_frozen if clip_model_frozen is not None else clip_model
        self.text_transformer = text_src.transformer
        self.text_ln_final    = text_src.ln_final
        self.text_transformer.apply(freeze_all_but_bn)
        self.register_buffer("text_pos_embed", text_src.positional_embedding.data.clone())
        self.register_buffer("text_proj",      text_src.text_projection.data.clone())

        prefix_photo  = getattr(cfg, "ctx_init",        "a photo of a")
        prefix_sketch = getattr(cfg, "ctx_init_sketch", "a sketch of a")
        n_ctx  = int(getattr(cfg, 'n_ctx', 3))
        ctx_dim = text_src.ln_final.weight.shape[0]   # 512 cho ViT-B/32

        # Khởi tạo ctx từ embedding của các token đầu trong prefix
        with torch.no_grad():
            ph_prefix_tok = _clip.tokenize([prefix_photo]).to(original_device)
            sk_prefix_tok = _clip.tokenize([prefix_sketch]).to(original_device)
            ph_prefix_emb = text_src.token_embedding(ph_prefix_tok).type(self.dtype)
            sk_prefix_emb = text_src.token_embedding(sk_prefix_tok).type(self.dtype)
            ctx_photo_init  = ph_prefix_emb[0, 1:1 + n_ctx, :].clone()   # (n_ctx, ctx_dim)
            ctx_sketch_init = sk_prefix_emb[0, 1:1 + n_ctx, :].clone()

        self.ctx_photo  = nn.Parameter(ctx_photo_init)    # learnable, (n_ctx, ctx_dim)
        self.ctx_sketch = nn.Parameter(ctx_sketch_init)   # learnable, (n_ctx, ctx_dim)
        self.dropout_ctx = nn.Dropout(p=0.1)

        # Tìm prefix_len: số token [SOS + words] trước EOS trong prefix
        eot_id = int(_clip.tokenize([""])[0, 1].item())   # EOS token id = 49407
        ph_prefix_len = int((ph_prefix_tok[0] == eot_id).nonzero()[0].item())
        sk_prefix_len = int((sk_prefix_tok[0] == eot_id).nonzero()[0].item())

        # Build prefix & suffix embedding cho mỗi class
        # Template: "<prefix> <X×n_ctx> <classname>."
        placeholder = " ".join(["X"] * n_ctx)
        ph_tmpl = [f"{prefix_photo} {placeholder} {n}.".replace("_", " ") for n in classnames]
        sk_tmpl = [f"{prefix_sketch} {placeholder} {n}.".replace("_", " ") for n in classnames]

        with torch.no_grad():
            ph_tok = _clip.tokenize(ph_tmpl).to(original_device)
            sk_tok = _clip.tokenize(sk_tmpl).to(original_device)
            ph_emb = text_src.token_embedding(ph_tok).type(self.dtype)   # (n_cls, 77, ctx_dim)
            sk_emb = text_src.token_embedding(sk_tok).type(self.dtype)

        self.register_buffer("text_prefix_photo",  ph_emb[:, :ph_prefix_len, :])
        self.register_buffer("text_suffix_photo",  ph_emb[:, ph_prefix_len + n_ctx:, :])
        self.register_buffer("text_prefix_sketch", sk_emb[:, :sk_prefix_len, :])
        self.register_buffer("text_suffix_sketch", sk_emb[:, sk_prefix_len + n_ctx:, :])
        self.register_buffer("text_tok_photo",  ph_tok)   # (n_cls, 77) — cho EOT lookup
        self.register_buffer("text_tok_sketch", sk_tok)

        print(f"Learnable text ctx: n_ctx={n_ctx}, ctx_dim={ctx_dim}, "
              f"ph_prefix_len={ph_prefix_len}, sk_prefix_len={sk_prefix_len}")

        # Learnable prompts cho HiCroPL backbone
        from src.clip.model import VisionTransformer_HiCroPL
        self._use_hicropl = isinstance(self.ph_encoder, VisionTransformer_HiCroPL)
        if self._use_hicropl:
            vision_ctx   = int(getattr(cfg, 'vision_ctx',   -1))
            if vision_ctx == -1:
                vision_ctx = int(getattr(cfg, 'n_ctx', 3))
            vision_depth = int(getattr(cfg, 'vision_depth', -1))
            if vision_depth == -1:
                vision_depth = int(getattr(cfg, 'n_prompts', 3))

            prompt_dim = self.ph_encoder.conv1.weight.shape[0]  # 768 cho ViT-B/32

            self.visual_prompt_photo  = nn.Parameter(torch.empty(vision_ctx, prompt_dim, dtype=self.dtype))
            self.visual_prompt_sketch = nn.Parameter(torch.empty(vision_ctx, prompt_dim, dtype=self.dtype))
            nn.init.normal_(self.visual_prompt_photo,  std=0.02)
            nn.init.normal_(self.visual_prompt_sketch, std=0.02)

            # Full-depth: mỗi encoder luôn có prompt ở tất cả n_vit layers.
            # vision_depth xác định bao nhiêu layer đầu của photo (và bấy nhiêu layer
            # cuối của sketch) là TRAINABLE; phần còn lại bị đóng băng (requires_grad=False).
            n_vit  = len(list(self.ph_encoder.transformer.resblocks))
            n_deep = n_vit - 1   # 11 cho ViT-B/32

            self.cross_prompts_photo  = nn.ParameterList(
                [nn.Parameter(torch.empty(vision_ctx, prompt_dim, dtype=self.dtype)) for _ in range(n_deep)]
            )
            self.cross_prompts_sketch = nn.ParameterList(
                [nn.Parameter(torch.empty(vision_ctx, prompt_dim, dtype=self.dtype)) for _ in range(n_deep)]
            )
            for p in list(self.cross_prompts_photo) + list(self.cross_prompts_sketch):
                nn.init.normal_(p, std=0.02)

            # Bật add_prompt cho tất cả layer 1..n_vit-1 trên cả hai encoder
            for enc in (self.ph_encoder, self.sk_encoder):
                enc.prompt_start_layer = 0
                for blk in enc.transformer.resblocks:
                    blk.prompt_start_layer = 0
                    if blk.i != 0:
                        blk.add_prompt = True

            # Đóng băng prompt của photo tại layers vision_depth..n_vit-1
            # cross_prompts_photo[i] dùng tại layer i+1
            # → freeze indices vision_depth-1 .. n_deep-1
            for i in range(vision_depth - 1, n_deep):
                self.cross_prompts_photo[i].requires_grad_(False)

            # Đóng băng prompt của sketch tại layers 0..vision_depth-1
            # → freeze visual_prompt_sketch (layer 0) + indices 0..vision_depth-2
            self.visual_prompt_sketch.requires_grad_(False)
            for i in range(vision_depth - 1):
                self.cross_prompts_sketch[i].requires_grad_(False)

            sk_start = n_vit - vision_depth
            n_ph_trainable = vision_depth          # shallow + cross[0..vision_depth-2]
            n_sk_trainable = n_vit - vision_depth  # cross[vision_depth-1..n_deep-1]
            print(f"HiCroPL full-depth prompts: n_vit={n_vit}, vision_ctx={vision_ctx}")
            print(f"  Photo  : layers 0–{n_vit-1} có prompt | trainable layers 0–{vision_depth-1} | frozen layers {vision_depth}–{n_vit-1}")
            print(f"  Sketch : layers 0–{n_vit-1} có prompt | frozen layers 0–{vision_depth-1}   | trainable layers {sk_start}–{n_vit-1}")

    def encode_text(self, modality):
        """Tính text features động với learnable ctx mỗi forward pass."""
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
        ctx_exp = ctx.unsqueeze(0).expand(n_cls, -1, -1)          # (n_cls, n_ctx, ctx_dim)
        prompts = torch.cat([prefix, ctx_exp, suffix], dim=1)      # (n_cls, 77, ctx_dim)

        x = prompts + self.text_pos_embed.type(self.dtype)
        x = x.permute(1, 0, 2)                                    # NLD → LND
        x = self.text_transformer(x)
        x = x.permute(1, 0, 2)                                    # LND → NLD
        x = self.text_ln_final(x).type(self.dtype)
        x = x[torch.arange(n_cls), tok.argmax(dim=-1)] @ self.text_proj
        return x / x.norm(dim=-1, keepdim=True)                    # (n_cls, 512)

    def encode_visual(self, x, modality):
        if modality == "photo":
            encoder = self.ph_encoder
            if self._use_hicropl:
                return encoder(x.type(self.dtype),
                               self.visual_prompt_photo,
                               list(self.cross_prompts_photo))
        else:
            encoder = self.sk_encoder
            if self._use_hicropl:
                return encoder(x.type(self.dtype),
                               self.visual_prompt_sketch,
                               list(self.cross_prompts_sketch))
        return encoder(x.type(self.dtype))

    def forward(self, x, classnames):
        sk_tensor    = x[0]
        photo_tensor = x[1]
        neg_tensor   = x[2]
        label        = x[5] if len(x) >= 6 else x[3]

        sketch_feat = self.encode_visual(sk_tensor,    "sketch")
        photo_feat  = self.encode_visual(photo_tensor, "photo")
        neg_feat    = self.encode_visual(neg_tensor,   "photo")

        sketch_feat = sketch_feat / sketch_feat.norm(dim=-1, keepdim=True)
        photo_feat  = photo_feat  / photo_feat.norm(dim=-1, keepdim=True)
        neg_feat    = neg_feat    / neg_feat.norm(dim=-1, keepdim=True)

        logit_scale   = self.logit_scale.exp()
        text_feat_photo  = self.encode_text("photo")
        text_feat_sketch = self.encode_text("sketch")
        logits_photo  = logit_scale * photo_feat  @ text_feat_photo.t()
        logits_sketch = logit_scale * sketch_feat @ text_feat_sketch.t()

        return (
            photo_feat, logits_photo,
            sketch_feat, logits_sketch,
            neg_feat, label,
        )


class HiCroPL_SBIR(pl.LightningModule):
    def __init__(self, cfg, args, classnames, model):
        super().__init__()
        self.cfg        = cfg
        self.args       = args
        self.classnames = classnames
        self.model      = model

        self.best_metric = 1e-3

        self.test_photo_features  = []
        self.test_sketch_features = []
        self.test_photo_labels    = []
        self.test_sketch_labels   = []

    def on_train_epoch_start(self):
        pass

    def configure_optimizers(self):
        """Adam — train LN + MHA in_proj + naked params của 2 visual encoder + HiCroPL prompts."""
        clip_params = (
            list(self.model.ph_encoder.parameters()) +
            list(self.model.sk_encoder.parameters()) +
            [self.model.logit_scale]
        )
        # Learnable text context + text encoder LN
        clip_params += [self.model.ctx_photo, self.model.ctx_sketch]
        clip_params += list(self.model.text_transformer.parameters())
        clip_params += list(self.model.text_ln_final.parameters())

        if getattr(self.model, '_use_hicropl', False):
            clip_params += [self.model.visual_prompt_photo, self.model.visual_prompt_sketch]
            clip_params += list(self.model.cross_prompts_photo)
            clip_params += list(self.model.cross_prompts_sketch)

        clip_trainable = sum(p.numel() for p in clip_params if p.requires_grad)
        self.print(f"Trainable params (ph_encoder + sk_encoder + logit_scale + prompts): {clip_trainable:,}")

        lr           = getattr(self.cfg, 'clip_LN_lr',   1e-5)
        weight_decay = getattr(self.cfg, 'weight_decay', 0.0)

        return torch.optim.Adam(
            [p for p in clip_params if p.requires_grad],
            lr=lr, weight_decay=weight_decay,
        )

    def training_step(self, batch, batch_idx):
        from src.losses_hicropl import loss_fn_hicropl
        features = self.model(batch, self.classnames)
        loss = loss_fn_hicropl(self.args, features)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss',       loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        return loss

    def extract_eval_features(self, tensor, modality):
        feat = self.model.encode_visual(tensor, modality)
        return feat / feat.norm(dim=-1, keepdim=True)

    def validation_step(self, batch, _, dataloader_idx=0):
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
        if not self.test_photo_features or not self.test_sketch_features:
            self.print("Warning: Missing features for validation. Skipping metrics.")
            return

        gallery_features = torch.cat(self.test_photo_features,  dim=0).to(self.device)
        query_features   = torch.cat(self.test_sketch_features, dim=0).to(self.device)
        all_photo_cat    = torch.cat(self.test_photo_labels,    dim=0).to(self.device)
        all_sketch_cat   = torch.cat(self.test_sketch_labels,   dim=0).to(self.device)

        similarity_matrix = query_features @ gallery_features.t()

        dataset = getattr(self.args, 'dataset', 'sketchy')
        if dataset in ("sketchy_2", "sketchy_ext"):
            map_k, p_k = 200, 200
        elif dataset == "quickdraw":
            map_k, p_k = 0, 200
        else:
            map_k, p_k = 0, 100

        n_g   = gallery_features.shape[0]
        ranks = torch.arange(1, n_g + 1, device=self.device, dtype=torch.float32)
        kk    = min(map_k, n_g) if map_k != 0 else n_g

        ap_lenient = torch.zeros(len(query_features), device=self.device)
        ap_all     = torch.zeros(len(query_features), device=self.device)
        ap_strict  = torch.zeros(len(query_features), device=self.device)
        precision  = torch.zeros(len(query_features), device=self.device)

        for idx in range(len(query_features)):
            category = all_sketch_cat[idx]
            sim      = similarity_matrix[idx]
            target   = (all_photo_cat == category)
            R        = target.sum().clamp(min=1).float()

            order   = torch.argsort(sim, descending=True)
            rel     = target[order].float()
            prec_at = torch.cumsum(rel, dim=0) / ranks

            hit_k            = (prec_at[:kk] * rel[:kk]).sum()
            ap_lenient[idx]  = hit_k / rel[:kk].sum().clamp(min=1)
            ap_strict[idx]   = hit_k / torch.minimum(R, torch.tensor(float(kk), device=self.device))
            ap_all[idx]      = (prec_at * rel).sum() / R
            precision[idx]   = rel[:min(p_k, n_g)].sum() / p_k

        m_lenient      = torch.mean(ap_lenient)
        m_all          = torch.mean(ap_all)
        m_strict       = torch.mean(ap_strict)
        mean_precision = torch.mean(precision)

        mAP = m_lenient
        self.log("mAP",         mAP,   on_step=False, on_epoch=True)
        self.log("val_mAP",     mAP,   on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_map_all", m_all, on_step=False, on_epoch=True)
        if map_k != 0:
            self.log(f"val_map_{map_k}",        m_lenient, on_step=False, on_epoch=True)
            self.log(f"val_map_{map_k}_strict",  m_strict,  on_step=False, on_epoch=True)
        self.log(f"P@{p_k}",    mean_precision, on_step=False, on_epoch=True)
        self.log(f"val_P@{p_k}", mean_precision, on_step=False, on_epoch=True)
        self.log(f"val_p_{p_k}", mean_precision, on_step=False, on_epoch=True)
        self.log("best_mAP",    self.best_metric, on_step=False, on_epoch=True, prog_bar=False)

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

    def test_step(self, batch, _, dataloader_idx=0):
        return self.validation_step(batch, _, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
