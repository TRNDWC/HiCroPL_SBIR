import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision

from src.clip import clip
from src.hicropl import (
    CrossModalPromptLearner,
    TextEncoder,
    VisualEncoder,
)
from src.hicropl_shared_text import SharedTextDualVisualPromptLearner

def freeze_model(m):
    m.requires_grad_(False)

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

def set_ln_to_train(m):
    """Set LayerNorm modules to train mode while keeping others in eval"""
    if isinstance(m, torch.nn.LayerNorm):
        m.train()



class CustomCLIP(nn.Module):
    """
    HiCroPL-SBIR Architecture Wrapper with Shared Text Prompts.
    
    Contains:
    - 1x SharedTextDualVisualPromptLearner (shared text + dual visual prompts)
    - 1x TextEncoder (shared, with deep prompt injection)
    - 1x VisualEncoder (shared, with deep prompt injection)
    - 1x Frozen CLIP (for consistency loss reference features)
    """

    def __init__(self, cfg, clip_model, clip_model_frozen):
        super().__init__()
        self.cfg = cfg
        self.clip_model = clip_model
        self.clip_model_frozen = clip_model_frozen
        
        # Freeze the base CLIP models (except LayerNorm which will be trainable)
        self.clip_model.apply(freeze_all_but_bn)
        self.clip_model_frozen.apply(freeze_model)  # Frozen model stays completely frozen
        
        self.dtype = clip_model.dtype
        self.logit_scale = clip_model.logit_scale
        
        # --- 1. Shared Text + Dual Visual Prompt Learner ---
        n_ctx = getattr(cfg, 'n_ctx', 4)
        prompt_depth = getattr(cfg, 'prompt_depth', 9)
        cross_layer = getattr(cfg, 'cross_layer', 4)
        ctx_init = getattr(cfg, 'ctx_init', "a photo or a sketch of a")
        
        print("Initializing Shared Text + Dual Visual Prompt Learner...")
        self.prompt_learner = SharedTextDualVisualPromptLearner(
            clip_model=clip_model,
            n_ctx=n_ctx,
            prompt_depth=prompt_depth,
            cross_layer=cross_layer,
            ctx_init=ctx_init,
            use_fp16=True if self.dtype == torch.float16 else False
        )

        # --- 2. Encoders with Deep Injection ---
        self.text_encoder = TextEncoder(clip_model)
        self.visual_encoder = VisualEncoder(clip_model)
        
        # Unfreeze LayerNorm in encoders for training (following HiCroPL practice)
        for name, param in self.text_encoder.named_parameters():
            if 'ln' in name.lower() or 'layernorm' in name.lower():
                param.requires_grad_(True)
        
        for name, param in self.visual_encoder.named_parameters():
            if 'ln' in name.lower() or 'layernorm' in name.lower():
                param.requires_grad_(True)
        
        # --- 3. Frozen Reference Model ---
        self.frozen_visual_encoder = clip_model_frozen.visual

    def forward(self, x, classnames):
        """
        Forward pass for training with augmentation support.
        x: batch from DataLoader with augmented images
        Format: [sk_tensor, img_tensor, neg_tensor, sk_aug_tensor, img_aug_tensor, label, filename]
        """
        sk_tensor, photo_tensor, neg_tensor, sk_aug_tensor, photo_aug_tensor, label = x[:6]
        
        # 1. Evaluate Shared Prompt Learner Once
        # Returns: (text_input, tok, first_v_p, deep_t, deep_v_p, first_v_s, deep_v_s)
        (
            text_input, tok, 
            first_v_p, deep_t, deep_v_p,
            first_v_s, deep_v_s
        ) = self.prompt_learner(classnames)
        
        # 2. Extract Text Features Once (shared for both photo and sketch)
        text_feat = self.text_encoder(text_input, tok, deep_t)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        # 3. Extract Visual Features with Prompts (Trainable Branch)
        # - Photo
        photo_feat = self.visual_encoder(photo_tensor, first_v_p, deep_v_p)
        photo_feat = photo_feat / photo_feat.norm(dim=-1, keepdim=True)
        
        # - Sketch
        sketch_feat = self.visual_encoder(sk_tensor, first_v_s, deep_v_s)
        sketch_feat = sketch_feat / sketch_feat.norm(dim=-1, keepdim=True)
        
        # - Negative Photo (Uses Photo Prompts)
        neg_feat = self.visual_encoder(neg_tensor, first_v_p, deep_v_p)
        neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
        
        # 4. Extract Augmented Visual Features WITHOUT Learnable Prompts
        # Use encoder with NO prompts (like frozen CLIP) for augmented images
        # This provides pure visual features without bidirectional knowledge flow
        photo_aug_feat = self.visual_encoder(photo_aug_tensor, None, None)
        photo_aug_feat = photo_aug_feat / photo_aug_feat.norm(dim=-1, keepdim=True)
        
        sketch_aug_feat = self.visual_encoder(sk_aug_tensor, None, None)
        sketch_aug_feat = sketch_aug_feat / sketch_aug_feat.norm(dim=-1, keepdim=True)
            
        # 5. Compute Logits (using shared text features)
        logit_scale = self.logit_scale.exp()
        logits_photo = logit_scale * photo_feat @ text_feat.t()
        logits_sketch = logit_scale * sketch_feat @ text_feat.t()
        
        # 6. Compute Logits for Augmented Images (using shared text features)
        logits_photo_aug = logit_scale * photo_aug_feat @ text_feat.t()
        logits_sketch_aug = logit_scale * sketch_aug_feat @ text_feat.t()
        
        return (
            photo_feat, logits_photo,
            sketch_feat, logits_sketch,
            neg_feat, label,
            photo_aug_feat, sketch_aug_feat,
            logits_photo_aug, logits_sketch_aug
        )
    
    def get_visual_prompts(self):
        """Return photo and sketch visual prompts for regularization loss.
        
        Returns:
            photo_prompts: list of L tensors [n_ctx, v_dim]
            sketch_prompts: list of L tensors [n_ctx, v_dim]
        """
        return (
            [p for p in self.prompt_learner.cross_prompts_visual_photo],
            [p for p in self.prompt_learner.cross_prompts_visual_sketch]
        )


class HiCroPL_SBIR(pl.LightningModule):
    def __init__(self, cfg, args, classnames, model):
        super().__init__()
        self.cfg = cfg
        self.args = args
        self.classnames = classnames
        self.model = model
        
        self.best_metric = 1e-3

        # Temporary buffer for metrics
        self.test_photo_features = []
        self.test_sketch_features = []
        self.test_photo_labels = []
        self.test_sketch_labels = []

        # Cache prompt outputs để tránh recompute mỗi validation batch
        self._cached_photo_prompts = None
        self._cached_sketch_prompts = None

    def on_train_epoch_start(self):
        # Ensure that clip models are in fully eval mode during training
        self.model.clip_model_frozen.eval()
        self.model.clip_model.eval()

    def configure_optimizers(self):
        clip_ln_params = []
        prompt_params = []
        seen_ids = set()
        
        for name, p in self.model.named_parameters():
            if p.requires_grad and id(p) not in seen_ids:
                seen_ids.add(id(p))
                if 'prompt_learner' in name:
                    prompt_params.append(p)
                else:
                    clip_ln_params.append(p)
        
        self.print(f"Number of trainable prompt params: {sum(p.numel() for p in prompt_params):,}")
        self.print(f"Number of trainable LayerNorm params: {sum(p.numel() for p in clip_ln_params):,}")
        
        prompt_lr = getattr(self.cfg, 'prompt_lr', 1e-5)
        clip_ln_lr = getattr(self.cfg, 'clip_LN_lr', 1e-5)
        weight_decay = getattr(self.cfg, 'weight_decay', 1e-4)

        param_groups = [{'params': prompt_params, 'lr': prompt_lr}]
        if clip_ln_params:
            param_groups.append({'params': clip_ln_params, 'lr': clip_ln_lr})

        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
        
        return optimizer

    def training_step(self, batch, batch_idx):
        from src.losses_hicropl import loss_fn_hicropl
        
        # Unpack batch and push to device happens in Lightning automatically, but we might need label explicitly
        features = self.model(batch, self.classnames)
        
        # Calculate custom loss (pass model for prompt alignment loss)
        loss = loss_fn_hicropl(self.args, features, model=self.model)
        
        # Log to TensorBoard (both step and epoch) but NOT to progress bar
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # Log to Progress Bar only the epoch average for cleaner output (set prog_bar=False as requested)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        
        return loss

    def on_validation_epoch_start(self):
        """Cache prompt outputs ONE TIME trước toàn bộ validation loop (giống GitHub CoPrompt)."""
        with torch.no_grad():
            # Call shared prompt learner once - returns (text_input, tok, first_v_p, deep_t, deep_v_p, first_v_s, deep_v_s)
            _, _, first_v_p, _, deep_v_p, first_v_s, deep_v_s = self.model.prompt_learner(self.classnames)
            
            self._cached_photo_prompts = (
                first_v_p.detach() if first_v_p is not None else None,
                [p.detach() for p in deep_v_p] if deep_v_p is not None else None,
            )
            self._cached_sketch_prompts = (
                first_v_s.detach() if first_v_s is not None else None,
                [p.detach() for p in deep_v_s] if deep_v_s is not None else None,
            )

    def extract_eval_features(self, tensor, modality):
        """Extract normalized visual features dùng CACHED prompts (không re-compute learner)."""
        cached = self._cached_photo_prompts if modality == 'photo' else self._cached_sketch_prompts
        first_visual_prompt, deeper_visual_prompts = cached
        with torch.no_grad():
            visual_features = self.model.visual_encoder(tensor, first_visual_prompt, deeper_visual_prompts)
            visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        return visual_features_norm

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
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

    def on_validation_epoch_end(self):
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

    # Reuse validation logic for testing
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
