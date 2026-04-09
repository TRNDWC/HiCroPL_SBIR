import copy
import numpy as np
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
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


class Adapter(nn.Module):
    """CoPrompt-style bottleneck adapter for feature refinement."""

    def __init__(self, c_in, reduction=4):
        super().__init__()
        hidden_dim = max(1, c_in // reduction)
        self.fc = nn.Sequential(
            nn.Linear(c_in, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fc(x)


class CustomCLIP(nn.Module):
    """
    HiCroPL-SBIR Architecture Wrapper.
    Sử dụng HiCroPLFeatureExtractor làm nòng cốt.
    """

    def __init__(self, cfg, clip_model, clip_model_frozen):
        super().__init__()
        self.cfg = cfg

        # 1. Distill branch (vanilla CLIP)
        self.clip_model_distill = clip_model_frozen
        freeze_model(self.clip_model_distill)
        self.distill_visual_encoder = self.clip_model_distill.visual

        # 2. Shared Logit Scale
        self.logit_scale = clip_model.logit_scale

        # 3. Prompted Branches (HiCroPL)
        clip_model.apply(freeze_all_but_bn)
        self.dtype = clip_model.dtype

        # Local deepcopies for per-modality branches (photo/sketch)
        clip_photo = copy.deepcopy(clip_model)
        clip_sketch = copy.deepcopy(clip_model)

        # -- Prompt Learners --
        print("Initializing Photo Prompt Learner...")
        self.prompt_learner_photo = CrossModalPromptLearner(
            cfg=cfg,
            clip_model=clip_photo,
            clip_model_distill=self.clip_model_distill
        )

        print("Initializing Sketch Prompt Learner...")
        self.prompt_learner_sketch = CrossModalPromptLearner(
            cfg=cfg,
            clip_model=clip_sketch,
            clip_model_distill=self.clip_model_distill
        )

        # -- Encoders --
        self.text_encoder_photo = TextEncoder(clip_photo)
        self.text_encoder_sketch = TextEncoder(clip_sketch)
        self.visual_encoder_photo = VisualEncoder(clip_photo)
        self.visual_encoder_sketch = VisualEncoder(clip_sketch)
        
        # -- Khởi tạo Feature Extractors --
        self.extractor_photo = HiCroPLFeatureExtractor(
            prompt_learner=self.prompt_learner_photo,
            text_encoder=self.text_encoder_photo,
            image_encoder=self.visual_encoder_photo,
            logit_scale=self.logit_scale,
            dtype=self.dtype
        )
        self.extractor_sketch = HiCroPLFeatureExtractor(
            prompt_learner=self.prompt_learner_sketch,
            text_encoder=self.text_encoder_sketch,
            image_encoder=self.visual_encoder_sketch,
            logit_scale=self.logit_scale,
            dtype=self.dtype
        )

        # --- 4. Shared Adapters ---
        adapter_reduction = getattr(cfg, 'adapter_reduction', 4)
        embed_dim = int(clip_photo.text_projection.shape[1])
        self.adapter_photo = Adapter(embed_dim, adapter_reduction).to(dtype=self.dtype)
        self.adapter_text = Adapter(embed_dim, adapter_reduction).to(dtype=self.dtype)
        self.use_adapter = bool(getattr(cfg, 'use_adapter', False))
        
        self.image_adapter_m = float(getattr(cfg, 'image_adapter_m', 0.1))
        self.text_adapter_m = float(getattr(cfg, 'text_adapter_m', 0.1))

        if not self.use_adapter:
            for p in self.adapter_photo.parameters():
                p.requires_grad_(False)
            for p in self.adapter_text.parameters():
                p.requires_grad_(False)
        
        adapter_param_count = (
            sum(p.numel() for p in self.adapter_photo.parameters() if p.requires_grad)
            + sum(p.numel() for p in self.adapter_text.parameters() if p.requires_grad)
        )
        print(f"Adapter enabled: {self.use_adapter} | trainable adapter params: {adapter_param_count:,}")
        print(f"Adapter mix ratio | image_adapter_m: {self.image_adapter_m:.3f}, text_adapter_m: {self.text_adapter_m:.3f}")

    def apply_adapter_residual(self, feat_prenorm, adapter, mix_ratio):
        """Adapter trước normalize — giống CoPrompt.
        feat_prenorm: feature chưa normalize cuối (raw hoặc residual_sum chưa norm).
        Đầu ra: feature đã normalize sau khi mix.
        """
        if not self.use_adapter:
            return feat_prenorm / feat_prenorm.norm(dim=-1, keepdim=True)
        x_a = adapter(feat_prenorm)                              # adapter trên feature thô
        feat = mix_ratio * x_a + (1 - mix_ratio) * feat_prenorm  # residual mix
        return feat / feat.norm(dim=-1, keepdim=True)            # normalize sau

    def forward(self, x, classnames):
        """
        Forward pass for training with augmentation support.
        x: batch from DataLoader with augmented images
        Format: [sk_tensor, img_tensor, neg_tensor, sk_aug_tensor, img_aug_tensor, label, filename]
        """
        sk_tensor, photo_tensor, neg_tensor, sk_aug_tensor, photo_aug_tensor, label = x[:6]
        
        # 1. Trích xuất Feature Cốt lõi thông qua Extractor
        out_p = self.extractor_photo(photo_tensor, classnames)
        out_s = self.extractor_sketch(sk_tensor, classnames)
        
        # Đặc trưng Negative lấy từ nhánh Photo
        out_neg = self.extractor_photo(neg_tensor, classnames)
        
        # 2. Visual features: Adapter trước normalize (giống CoPrompt)
        #    dùng prenorm = (norm_learned + fixed) chưa normalize cuối
        photo_feat  = self.apply_adapter_residual(out_p["image_features_prenorm"],   self.adapter_photo, self.image_adapter_m)
        sketch_feat = self.apply_adapter_residual(out_s["image_features_prenorm"],   self.adapter_photo, self.image_adapter_m)
        neg_feat    = self.apply_adapter_residual(out_neg["image_features_prenorm"], self.adapter_photo, self.image_adapter_m)
        
        # Text features: qua Adapter (trừ nhánh aug)
        text_feat_photo  = self.apply_adapter_residual(out_p["text_features_prenorm"],  self.adapter_text, self.text_adapter_m)
        text_feat_sketch = self.apply_adapter_residual(out_s["text_features_prenorm"],  self.adapter_text, self.text_adapter_m)

        # Trích xuất Fixed reference targets
        photo_feat_fixed = out_p["image_features_fixed"]
        sketch_feat_fixed = out_s["image_features_fixed"]
        text_feat_fixed_photo = out_p["text_features_fixed"]
        text_feat_fixed_sketch = out_s["text_features_fixed"]

        # 3. Aug Visual Features: shared distill branch (Không qua Adapter)
        photo_aug_feat = self.distill_visual_encoder(photo_aug_tensor.type(self.dtype))
        photo_aug_feat = photo_aug_feat / photo_aug_feat.norm(dim=-1, keepdim=True)

        sketch_aug_feat = self.distill_visual_encoder(sk_aug_tensor.type(self.dtype))
        sketch_aug_feat = sketch_aug_feat / sketch_aug_feat.norm(dim=-1, keepdim=True)
            
        # 4. Compute Logits
        logit_scale = out_p["logit_scale"]
        logits_photo = logit_scale * photo_feat @ text_feat_photo.t()
        logits_sketch = logit_scale * sketch_feat @ text_feat_sketch.t()
        
        # 5. Compute Logits for Augmented Images
        logits_photo_aug = logit_scale * photo_aug_feat @ text_feat_photo.t()
        logits_sketch_aug = logit_scale * sketch_aug_feat @ text_feat_sketch.t()
        
        return (
            photo_feat, logits_photo,
            sketch_feat, logits_sketch,
            neg_feat, label,
            photo_aug_feat, sketch_aug_feat,
            logits_photo_aug, logits_sketch_aug,
            text_feat_photo, text_feat_sketch,
            text_feat_fixed_photo, text_feat_fixed_sketch,
            photo_feat_fixed, sketch_feat_fixed,
        )


class HiCroPL_SBIR(pl.LightningModule):
    def __init__(self, cfg, args, classnames, model, text_proto_matrix=None):
        super().__init__()
        self.cfg = cfg
        self.args = args
        self.classnames = classnames
        self.model = model
        # Precomputed frozen text prototypes [n_seen, d] for structural loss.
        # Set via compute_text_prototypes() in the entrypoint, or None to fall
        # back to per-batch text_feat_fixed values.
        self.text_proto_matrix = text_proto_matrix
        
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
        self.model.clip_model_distill.eval()

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

        adapter_params = []
        if getattr(self.model, 'use_adapter', False):
            add_unique_params(self.model.adapter_photo.parameters(), adapter_params, seen_ids)
            add_unique_params(self.model.adapter_text.parameters(), adapter_params, seen_ids)

        extra_trainable_params = []
        for _, p in self.model.named_parameters():
            if p.requires_grad and id(p) not in seen_ids:
                seen_ids.add(id(p))
                extra_trainable_params.append(p)

        non_prompt_params = ln_params + adapter_params + extra_trainable_params

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
        loss = loss_fn_hicropl(self.args, features,
                               text_proto_matrix=self.text_proto_matrix)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        
        return loss

    def extract_eval_features(self, tensor, modality):
        """Extract visual features: Adapter trước normalize (giống CoPrompt)"""
        extractor = self.model.extractor_photo if modality == 'photo' else self.model.extractor_sketch
        out = extractor(tensor, self.classnames)
        # dùng prenorm làm input cho adapter, normalize sau
        return self.model.apply_adapter_residual(
            out["image_features_prenorm"], self.model.adapter_photo, self.model.image_adapter_m
        )

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
            self.test_sketch_features.append(sketch_feat.detach()) 
            self.test_sketch_labels.append(label.detach())
        elif dataloader_idx == 1:
            photo_feat = self.extract_eval_features(tensor, modality='photo')
            self.test_photo_features.append(photo_feat.detach())   
            self.test_photo_labels.append(label.detach())

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
            target_buckets[cat_idx]['features'].append(feat.detach())  
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

        gallery_features = torch.cat(self.test_photo_features, dim=0)   
        query_features   = torch.cat(self.test_sketch_features, dim=0)  
        
        all_photo_category  = torch.cat(self.test_photo_labels, dim=0)  
        all_sketch_category = torch.cat(self.test_sketch_labels, dim=0) 

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

        ap        = torch.zeros(len(query_features), device=self.device)
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

        mAP            = torch.mean(ap)
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
        sim_matrix = sketch_feats @ photo_feats.t()  
        distance_matrix = 1.0 - sim_matrix
        
        N_sk = len(sketch_feats)
        ranks = torch.zeros(N_sk, device=sketch_feats.device)
        
        for i in range(N_sk):
            sketch_base = sketch_base_names[i]
            try:
                gt_idx = photo_base_names.index(sketch_base)
            except ValueError:
                ranks[i] = len(photo_base_names) + 1
                continue
            
            distances = distance_matrix[i]
            gt_distance = distances[gt_idx]
            rank = (distances <= gt_distance).sum()
            ranks[i] = rank
        
        return ranks

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
