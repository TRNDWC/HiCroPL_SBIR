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

        # Pre-compute text features một lần với template cố định, lưu vào buffer.
        # Dùng clip_model_frozen (hoặc clip_model) fully frozen.
        text_src = clip_model_frozen if clip_model_frozen is not None else clip_model
        ctx_photo  = getattr(cfg, "ctx_init",        "a photo of a")
        ctx_sketch = getattr(cfg, "ctx_init_sketch", "a sketch of a")

        with torch.no_grad():
            ph_templates = [f"{ctx_photo} {n}.".replace("_", " ")  for n in classnames]
            sk_templates = [f"{ctx_sketch} {n}.".replace("_", " ") for n in classnames]

            ph_tok = _clip.tokenize(ph_templates).to(original_device)
            sk_tok = _clip.tokenize(sk_templates).to(original_device)

            ph_feats = text_src.encode_text(ph_tok).type(self.dtype)
            sk_feats = text_src.encode_text(sk_tok).type(self.dtype)

            ph_feats = ph_feats / ph_feats.norm(dim=-1, keepdim=True)
            sk_feats = sk_feats / sk_feats.norm(dim=-1, keepdim=True)

        self.register_buffer("text_feat_photo",  ph_feats)   # (n_cls, 512)
        self.register_buffer("text_feat_sketch", sk_feats)   # (n_cls, 512)

        print(f"Text features precomputed: photo {ph_feats.shape}, sketch {sk_feats.shape}")

        # Frozen distillation model — dùng list wrapper để PyTorch không register params
        self._distill = [clip_model_frozen if clip_model_frozen is not None else clip_model]

    def encode_visual(self, x, modality):
        encoder = self.ph_encoder if modality == "photo" else self.sk_encoder
        return encoder(x.type(self.dtype))   # không có prompt

    def forward(self, x, classnames):
        sk_tensor      = x[0]
        photo_tensor   = x[1]
        neg_tensor     = x[2]
        sk_aug_tensor  = x[3]
        img_aug_tensor = x[4]
        label          = x[5] if len(x) >= 6 else x[3]

        sketch_feat = self.encode_visual(sk_tensor,    "sketch")
        photo_feat  = self.encode_visual(photo_tensor, "photo")
        neg_feat    = self.encode_visual(neg_tensor,   "photo")

        sketch_feat = sketch_feat / sketch_feat.norm(dim=-1, keepdim=True)
        photo_feat  = photo_feat  / photo_feat.norm(dim=-1, keepdim=True)
        neg_feat    = neg_feat    / neg_feat.norm(dim=-1, keepdim=True)

        logit_scale   = self.logit_scale.exp()
        logits_photo  = logit_scale * photo_feat  @ self.text_feat_photo.t()
        logits_sketch = logit_scale * sketch_feat @ self.text_feat_sketch.t()

        with torch.no_grad():
            distill = self._distill[0]
            photo_aug_feat = distill.encode_image(img_aug_tensor.type(self.dtype))
            sk_aug_feat    = distill.encode_image(sk_aug_tensor.type(self.dtype))

        return (
            photo_feat, logits_photo,
            sketch_feat, logits_sketch,
            neg_feat, label,
            photo_aug_feat, sk_aug_feat,
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
        """Adam — chỉ train LN + MHA in_proj + naked params của 2 visual encoder."""
        clip_params = (
            list(self.model.ph_encoder.parameters()) +
            list(self.model.sk_encoder.parameters()) +
            [self.model.logit_scale]
        )
        clip_trainable = sum(p.numel() for p in clip_params if p.requires_grad)
        self.print(f"Trainable params (ph_encoder + sk_encoder + logit_scale): {clip_trainable:,}")

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
