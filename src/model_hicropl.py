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
    
        Contains:
        - 2x CrossModalPromptLearner (one for Photo, one for Sketch)
        - 2x TextEncoder (Photo/Sketch, with deep prompt injection)
        - 2x prompted visual encoders (photo/sketch)
        - 1x shared distill visual branch for augmentation views
        - Optional shared feature adapter (disabled by default for ablation)
        - Distill CLIP branch with LayerNorm trainable
    """

    def __init__(self, cfg, clip_model, clip_model_frozen):
        super().__init__()
        self.cfg = cfg

        # Build prompted backbones and one shared distill branch (CoPrompt-style).
        self.clip_model_photo = copy.deepcopy(clip_model)
        self.clip_model_sketch = copy.deepcopy(clip_model)
        self.clip_model_distill = clip_model_frozen
        
        # Freeze non-LN params and keep LayerNorm trainable across all encoder branches.
        self.clip_model_photo.apply(freeze_all_but_bn)
        self.clip_model_sketch.apply(freeze_all_but_bn)
        self.clip_model_distill.apply(freeze_all_but_bn)
        
        self.dtype = self.clip_model_photo.dtype
        
        # --- 1. Dual Prompt Learners ---
        n_ctx = getattr(cfg, 'n_ctx', 4)
        prompt_depth = getattr(cfg, 'prompt_depth', 9)
        cross_layer = getattr(cfg, 'cross_layer', 4)
        ctx_init_photo = getattr(cfg, 'ctx_init', "a photo or a sketch of a")
        ctx_init_sketch = getattr(cfg, 'ctx_init_sketch', "a photo or a sketch of a")
        
        print("Initializing Photo Prompt Learner...")
        self.prompt_learner_photo = CrossModalPromptLearner(
            clip_model=self.clip_model_photo,
            n_ctx=n_ctx,
            prompt_depth=prompt_depth,
            cross_layer=cross_layer,
            ctx_init=ctx_init_photo,
            use_fp16=True if self.dtype == torch.float16 else False
        )
        
        print("Initializing Sketch Prompt Learner...")
        self.prompt_learner_sketch = CrossModalPromptLearner(
            clip_model=self.clip_model_sketch,
            n_ctx=n_ctx,
            prompt_depth=prompt_depth,
            cross_layer=cross_layer,
            ctx_init=ctx_init_sketch,
            use_fp16=True if self.dtype == torch.float16 else False
        )

        # --- 2. Encoders with Deep Injection (A/B/C/D prompted branches) ---
        self.text_encoder_photo = TextEncoder(self.clip_model_photo)
        self.text_encoder_sketch = TextEncoder(self.clip_model_sketch)
        self.visual_encoder_photo = VisualEncoder(self.clip_model_photo)
        self.visual_encoder_sketch = VisualEncoder(self.clip_model_sketch)
        # Aug views use shared distill visual encoder (no HiCroPL prompts).

        # --- 2.1 Shared adapters (CoPrompt-style, always-on) ---
        adapter_reduction = getattr(cfg, 'adapter_reduction', 4)
        embed_dim = int(self.clip_model_photo.text_projection.shape[1])
        self.adapter_photo = Adapter(embed_dim, adapter_reduction).to(dtype=self.dtype)
        self.adapter_text = Adapter(embed_dim, adapter_reduction).to(dtype=self.dtype)
        
        self.image_adapter_m = float(getattr(cfg, 'image_adapter_m', getattr(cfg, 'visual_adapter_m', 0.1)))
        self.text_adapter_m = float(getattr(cfg, 'text_adapter_m', 0.1))
        
        adapter_param_count = (
            sum(p.numel() for p in self.adapter_photo.parameters() if p.requires_grad)
            + sum(p.numel() for p in self.adapter_text.parameters() if p.requires_grad)
        )
        print(f"Adapters ALWAYS enabled | trainable adapter params: {adapter_param_count:,}")
        print(f"Adapter mix ratio | image_adapter_m: {self.image_adapter_m:.3f}, text_adapter_m: {self.text_adapter_m:.3f}")
        
        # LN remains trainable on prompted and distill branches via freeze_all_but_bn.
        
        # --- 3. Shared Distill Visual Branch ---
        self.distill_visual_encoder = self.clip_model_distill.visual

    def forward(self, x, classnames):
        """
        Forward pass for training with augmentation support.
        x: batch from DataLoader with augmented images
        Format: [sk_tensor, img_tensor, neg_tensor, sk_aug_tensor, img_aug_tensor, label, filename]
        """
        sk_tensor, photo_tensor, neg_tensor, sk_aug_tensor, photo_aug_tensor, label = x[:6]
        
        # 1. Evaluate Prompt Learners Once
        (
            text_input_p, tok_p, first_v_p, deep_t_p, deep_v_p
        ) = self.prompt_learner_photo(classnames)
        
        (
            text_input_s, tok_s, first_v_s, deep_t_s, deep_v_s
        ) = self.prompt_learner_sketch(classnames)

        # 1.1 Distill reference targets (no gradients)
        with torch.no_grad():
            text_feat_fixed_photo = self.clip_model_distill.encode_text(tok_p)
            text_feat_fixed_photo = text_feat_fixed_photo / text_feat_fixed_photo.norm(dim=-1, keepdim=True)

            text_feat_fixed_sketch = self.clip_model_distill.encode_text(tok_s)
            text_feat_fixed_sketch = text_feat_fixed_sketch / text_feat_fixed_sketch.norm(dim=-1, keepdim=True)

            photo_feat_fixed = self.distill_visual_encoder(photo_tensor.type(self.dtype))
            photo_feat_fixed = photo_feat_fixed / photo_feat_fixed.norm(dim=-1, keepdim=True)

            sketch_feat_fixed = self.distill_visual_encoder(sk_tensor.type(self.dtype))
            sketch_feat_fixed = sketch_feat_fixed / sketch_feat_fixed.norm(dim=-1, keepdim=True)

        # 2. Extract text features and apply adapter residual mixing (A/B).
        text_feat_photo = self.text_encoder_photo(text_input_p, tok_p, deep_t_p)
        x_b = self.adapter_text(text_feat_photo)
        text_feat_photo = self.text_adapter_m * x_b + (1 - self.text_adapter_m) * text_feat_photo
        text_feat_photo = text_feat_photo / text_feat_photo.norm(dim=-1, keepdim=True)
        
        text_feat_sketch = self.text_encoder_sketch(text_input_s, tok_s, deep_t_s)
        x_b = self.adapter_text(text_feat_sketch)
        text_feat_sketch = self.text_adapter_m * x_b + (1 - self.text_adapter_m) * text_feat_sketch
        text_feat_sketch = text_feat_sketch / text_feat_sketch.norm(dim=-1, keepdim=True)
        
        # 3. Aug Visual Features: shared distill branch with LN trainable, no HiCroPL prompts.
        photo_aug_feat = self.distill_visual_encoder(photo_aug_tensor.type(self.dtype))
        photo_aug_feat = photo_aug_feat / photo_aug_feat.norm(dim=-1, keepdim=True)

        sketch_aug_feat = self.distill_visual_encoder(sk_aug_tensor.type(self.dtype))
        sketch_aug_feat = sketch_aug_feat / sketch_aug_feat.norm(dim=-1, keepdim=True)

        # 4. Extract visual features and apply adapter residual mixing (C/D).
        # - Photo (C)
        photo_feat = self.visual_encoder_photo(photo_tensor.type(self.dtype), first_v_p, deep_v_p)
        x_a = self.adapter_photo(photo_feat)
        photo_feat = self.image_adapter_m * x_a + (1 - self.image_adapter_m) * photo_feat
        photo_feat = photo_feat / photo_feat.norm(dim=-1, keepdim=True)
        
        # - Sketch (D)
        sketch_feat = self.visual_encoder_sketch(sk_tensor, first_v_s, deep_v_s)
        x_a = self.adapter_photo(sketch_feat)
        sketch_feat = self.image_adapter_m * x_a + (1 - self.image_adapter_m) * sketch_feat
        sketch_feat = sketch_feat / sketch_feat.norm(dim=-1, keepdim=True)
        
        # - Negative Photo (Uses Photo Prompts)
        neg_feat = self.visual_encoder_photo(neg_tensor, first_v_p, deep_v_p)
        neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
            
        # 5. Compute Logits
        logit_scale_photo = self.clip_model_photo.logit_scale.exp()
        logit_scale_sketch = self.clip_model_sketch.logit_scale.exp()
        logits_photo = logit_scale_photo * photo_feat @ text_feat_photo.t()
        logits_sketch = logit_scale_sketch * sketch_feat @ text_feat_sketch.t()
        
        # 6. Compute Logits for Augmented Images
        logits_photo_aug = logit_scale_photo * photo_aug_feat @ text_feat_photo.t()
        logits_sketch_aug = logit_scale_sketch * sketch_aug_feat @ text_feat_sketch.t()
        
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
    def __init__(self, cfg, args, classnames, model):
        super().__init__()
        self.cfg = cfg
        self.args = args
        self.classnames = classnames
        self.model = model
        
        self.best_metric = 1e-3

        # ============ NEW: Eval mode flag ============
        self.eval_mode = getattr(args, 'eval_mode', 'category')

        # Temporary buffer for metrics
        self.test_photo_features = []
        self.test_sketch_features = []
        self.test_photo_labels = []
        self.test_sketch_labels = []

         # ============ NEW: Fine-grained buffers (per-category) ============
        from collections import defaultdict
        self.fg_sketch_buckets = defaultdict(lambda: {
            'features': [],    # List of tensors
            'filenames': [],   # List of filenames
            'base_names': []   # List of base names
        })
        self.fg_photo_buckets = defaultdict(lambda: {
            'features': [],
            'filenames': [],
            'base_names': []
        })

        # Cache prompt outputs để tránh recompute mỗi validation batch
        self._cached_photo_prompts = None
        self._cached_sketch_prompts = None

    def on_train_epoch_start(self):
        # Ensure that clip models are in fully eval mode during training
        self.model.clip_model_photo.eval()
        self.model.clip_model_sketch.eval()
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
        ln_backbones = [
            self.model.clip_model_photo,
            self.model.clip_model_sketch,
            self.model.clip_model_distill,
        ]
        for backbone in ln_backbones:
            for module in backbone.modules():
                if isinstance(module, torch.nn.LayerNorm):
                    add_unique_params(module.parameters(recurse=False), ln_params, seen_ids)

        adapter_params = []
        add_unique_params(self.model.adapter_photo.parameters(), adapter_params, seen_ids)
        add_unique_params(self.model.adapter_text.parameters(), adapter_params, seen_ids)
        adapter_param_total = sum(p.numel() for p in adapter_params)
        if adapter_param_total == 0:
            raise RuntimeError("Adapters always-on but no adapter parameters were collected by optimizer")

        extra_trainable_params = []
        for _, p in self.model.named_parameters():
            if p.requires_grad and id(p) not in seen_ids:
                seen_ids.add(id(p))
                extra_trainable_params.append(p)

        non_prompt_params = ln_params + adapter_params + extra_trainable_params

        self.print(f"Number of trainable prompt params: {sum(p.numel() for p in prompt_params):,}")
        self.print(f"Number of trainable LayerNorm params: {sum(p.numel() for p in ln_params):,}")
        self.print(f"Number of trainable adapter params: {adapter_param_total:,}")
        if extra_trainable_params:
            self.print(f"Warning: Found {len(extra_trainable_params)} unexpected trainable tensors outside prompt/LN/adapter; including them in optimizer.")
        self.print(f"Total trainable non-prompt params: {sum(p.numel() for p in non_prompt_params):,}")
        
        prompt_lr = getattr(self.cfg, 'prompt_lr', 1e-5)
        clip_ln_lr = getattr(self.cfg, 'clip_LN_lr', 1e-5)
        weight_decay = getattr(self.cfg, 'weight_decay', 1e-4)

        param_groups = [{'params': prompt_params, 'lr': prompt_lr}]
        if non_prompt_params:
            param_groups.append({'params': non_prompt_params, 'lr': clip_ln_lr})

        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
        
        return optimizer

    def training_step(self, batch, batch_idx):
        from src.losses_hicropl import loss_fn_hicropl
        
        # Unpack batch and push to device happens in Lightning automatically, but we might need label explicitly
        features = self.model(batch, self.classnames)
        
        # Calculate custom loss
        loss = loss_fn_hicropl(self.args, features)
        
        # Log to TensorBoard (both step and epoch) but NOT to progress bar
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # Log to Progress Bar only the epoch average for cleaner output (set prog_bar=False as requested)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        
        return loss

    def on_validation_epoch_start(self):
        """Cache prompt outputs ONE TIME trước toàn bộ validation loop (giống GitHub CoPrompt)."""
        with torch.no_grad():
            _, _, first_v_p, _, deep_v_p = self.model.prompt_learner_photo(self.classnames)
            self._cached_photo_prompts = (
                first_v_p.detach() if first_v_p is not None else None,
                [p.detach() for p in deep_v_p] if deep_v_p is not None else None,
            )
            _, _, first_v_s, _, deep_v_s = self.model.prompt_learner_sketch(self.classnames)
            self._cached_sketch_prompts = (
                first_v_s.detach() if first_v_s is not None else None,
                [p.detach() for p in deep_v_s] if deep_v_s is not None else None,
            )

    def extract_eval_features(self, tensor, modality):
        """Extract normalized visual features dùng CACHED prompts (không re-compute learner)."""
        cached = self._cached_photo_prompts if modality == 'photo' else self._cached_sketch_prompts
        first_visual_prompt, deeper_visual_prompts = cached
        if modality == 'photo':
            visual_encoder = self.model.visual_encoder_photo
        else:
            visual_encoder = self.model.visual_encoder_sketch
        with torch.no_grad():
            visual_features = visual_encoder(tensor, first_visual_prompt, deeper_visual_prompts)
            x_a = self.model.adapter_photo(visual_features)
            visual_features = self.model.image_adapter_m * x_a + (1 - self.model.image_adapter_m) * visual_features
            visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        return visual_features_norm

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.eval_mode == 'fine_grained':
            return self._validation_step_fg(batch, batch_idx, dataloader_idx)
        else:
            return self._validation_step_category(batch, batch_idx, dataloader_idx)

    def _validation_step_category(self, batch, batch_idx, dataloader_idx=0):
        # Depending on how ValidDataset yields, it's typically (image_tensor, label)
        # We use dataloader_idx to determine modality: 0 for sketch, 1 for photo
        if len(batch) == 3:
            tensor, label, type_data = batch
        else:
            tensor, label = batch
            type_data = None
            
        if dataloader_idx == 0:
            sketch_feat = self.extract_eval_features(tensor, modality='sketch')
            self.test_sketch_features.append(sketch_feat.detach()) # Keep on GPU
            self.test_sketch_labels.append(label.detach())
        elif dataloader_idx == 1:
            photo_feat = self.extract_eval_features(tensor, modality='photo')
            self.test_photo_features.append(photo_feat.detach())   # Keep on GPU
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
            target_buckets[cat_idx]['features'].append(feat.detach())  # Keep on GPU
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

        # Ghép features & labels
        gallery_features = torch.cat(self.test_photo_features, dim=0)   # [N_g, d]
        query_features   = torch.cat(self.test_sketch_features, dim=0)  # [N_q, d]
        
        all_photo_category  = torch.cat(self.test_photo_labels, dim=0)  # [N_g]
        all_sketch_category = torch.cat(self.test_sketch_labels, dim=0) # [N_q]

        # Tính toán ma trận tương quan trên GPU dùng Matrix Multiplication
        # Vì features đã được normalize, sim = dot product
        similarity_matrix = query_features @ gallery_features.t()       # [N_q, N_g]

        # Xác định top-k theo dataset
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
            distance = similarity_matrix[idx] # Scores on GPU

            # Target mask
            target = (all_photo_category == category)

            if map_k != 0:
                top_k_actual = min(map_k, len(gallery_features))
                ap[idx] = retrieval_average_precision(distance, target, top_k=top_k_actual)
            else:
                ap[idx] = retrieval_average_precision(distance, target)

            precision[idx] = retrieval_precision(distance, target, top_k=p_k)

        mAP            = torch.mean(ap)
        mean_precision = torch.mean(precision)

        # CoPrompt-compatible primary metric names
        self.log("mAP", mAP, on_step=False, on_epoch=True)
        self.log(f"P@{p_k}", mean_precision, on_step=False, on_epoch=True)

        self.log("val_mAP", mAP, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"val_P@{p_k}", mean_precision, on_step=False, on_epoch=True)
        # Log best metric (internally) without cluttering the progress bar
        self.log("best_mAP", self.best_metric, on_step=False, on_epoch=True, prog_bar=False)

        # Thêm alias key để ModelCheckpoint có thể monitor đúng tên
        if map_k != 0:
            self.log(f"val_map_{map_k}", mAP, on_step=False, on_epoch=True)
        else:
            self.log("val_map_all", mAP, on_step=False, on_epoch=True)
        self.log(f"val_p_{p_k}", mean_precision, on_step=False, on_epoch=True)

        # Track best mAP (giống GitHub)
        if self.global_step > 0:
            self.best_metric = self.best_metric if (self.best_metric > mAP.item()) else mAP.item()

        if map_k != 0:
            self.print('mAP@{}: {:.4f}, P@{}: {:.4f}, Best mAP: {:.4f}'.format(
                map_k, mAP.item(), p_k, mean_precision.item(), self.best_metric))
        else:
            self.print('mAP@all: {:.4f}, P@{}: {:.4f}, Best mAP: {:.4f}'.format(
                mAP.item(), p_k, mean_precision.item(), self.best_metric))

        # In train_loss (giống GitHub)
        train_loss = self.trainer.callback_metrics.get("train_loss", None)
        if train_loss is not None:
            self.print(f"Train loss (epoch avg): {train_loss.item():.6f}")

        # Clear buffers
        self.test_photo_features.clear()
        self.test_sketch_features.clear()
        self.test_photo_labels.clear()
        self.test_sketch_labels.clear()


    def _on_validation_epoch_end_fine_grained(self):
        """Fine-grained evaluation - compute Acc@k."""
        from src_fg.utils_fg import compute_rank_based_accuracy
        
        if len(self.fg_sketch_buckets) == 0 or len(self.fg_photo_buckets) == 0:
            self.print("Warning: No fine-grained data collected. Skipping FG metrics.")
            return
        
        all_ranks = []
        
        # Process each category
        for category_idx in self.fg_sketch_buckets.keys():
            if category_idx not in self.fg_photo_buckets:
                continue
            
            sketch_bucket = self.fg_sketch_buckets[category_idx]
            photo_bucket = self.fg_photo_buckets[category_idx]
            
            if len(sketch_bucket['features']) == 0 or len(photo_bucket['features']) == 0:
                continue
            
            # Stack features
            sketch_feats = torch.stack(sketch_bucket['features'])  # [N_sk, d]
            photo_feats = torch.stack(photo_bucket['features'])    # [N_ph, d]
            
            # Compute ranks for this category
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
        
        # Concatenate all ranks
        all_ranks_tensor = torch.cat(all_ranks)  # [total_sketches]
        
        # Compute accuracies
        result = compute_rank_based_accuracy(all_ranks_tensor, top_k_list=[1, 5, 10])
        
        acc1 = result['acc@1']
        acc5 = result['acc@5']
        acc10 = result['acc@10']

        # CoPrompt FG-style aliases
        top1 = acc1
        top5 = acc5
        
        # Log metrics
        self.log('fg_acc@1', acc1, on_epoch=True, prog_bar=True)
        self.log('fg_acc@5', acc5, on_epoch=True, prog_bar=True)
        self.log('fg_acc@10', acc10, on_epoch=True, prog_bar=True)
        self.log('top1', top1, on_epoch=True, prog_bar=True)
        self.log('top5', top5, on_epoch=True, prog_bar=True)
        
        # Update best metric (track acc@1 as main metric)
        if self.global_step > 0:
            self.best_metric = max(self.best_metric, acc1)
        self.log('best_fg_acc@1', self.best_metric, on_epoch=True, prog_bar=False)
        
        self.print(f'top1: {top1:.4f}, top5: {top5:.4f}, acc@10: {acc10:.4f}, Best: {self.best_metric:.4f}')
        
        # Clear buckets
        self.fg_sketch_buckets.clear()
        self.fg_photo_buckets.clear()


    def _compute_per_category_rank(self, sketch_feats, sketch_base_names, photo_feats, photo_base_names):
        """Compute rank of the correct photo for each sketch in a category."""
        sim_matrix = sketch_feats @ photo_feats.t()  # [N_sk, N_ph]
    
        # Convert to distance (lower = better)
        distance_matrix = 1.0 - sim_matrix
        
        # Compute ranks
        N_sk = len(sketch_feats)
        ranks = torch.zeros(N_sk, device=sketch_feats.device)
        
        for i in range(N_sk):
            sketch_base = sketch_base_names[i]
            
            # Find ground-truth photo index
            try:
                gt_idx = photo_base_names.index(sketch_base)
            except ValueError:
                # No matching photo - assign worst rank
                ranks[i] = len(photo_base_names) + 1
                continue
            
            # Get distances for this sketch
            distances = distance_matrix[i]
            
            # Ground-truth distance
            gt_distance = distances[gt_idx]
            
            # Rank = count of photos with distance <= gt_distance
            rank = (distances <= gt_distance).sum()
            ranks[i] = rank
        
        return ranks

    # Reuse validation logic for testing
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
