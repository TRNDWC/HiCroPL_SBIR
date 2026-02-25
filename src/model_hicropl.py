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
    HiCroPL-SBIR Architecture Wrapper.
    
    Contains:
    - 2x CrossModalPromptLearner (one for Photo, one for Sketch)
    - 1x TextEncoder (shared, with deep prompt injection)
    - 1x VisualEncoder (shared, with deep prompt injection)
    - 1x Frozen CLIP (for consistency loss reference features)
    """

    def __init__(self, cfg, clip_model, clip_model_frozen):
        super().__init__()
        self.cfg = cfg
        self.clip_model = clip_model
        self.clip_model_frozen = clip_model_frozen
        
        # Freeze the base CLIP models according to paper
        self.clip_model.apply(freeze_all_but_bn)
        self.clip_model_frozen.apply(freeze_model)
        
        self.dtype = clip_model.dtype
        self.logit_scale = clip_model.logit_scale
        
        # --- 1. Dual Prompt Learners ---
        n_ctx = getattr(cfg, 'n_ctx', 4)
        prompt_depth = getattr(cfg, 'prompt_depth', 9)
        cross_layer = getattr(cfg, 'cross_layer', 4)
        ctx_init = getattr(cfg, 'ctx_init', "a photo of a")
        
        print("Initializing Photo Prompt Learner...")
        self.prompt_learner_photo = CrossModalPromptLearner(
            clip_model=clip_model,
            n_ctx=n_ctx,
            prompt_depth=prompt_depth,
            cross_layer=cross_layer,
            ctx_init=ctx_init,
            use_fp16=True if self.dtype == torch.float16 else False
        )
        
        print("Initializing Sketch Prompt Learner...")
        self.prompt_learner_sketch = CrossModalPromptLearner(
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
        
        # --- 3. Frozen Reference Model ---
        self.frozen_visual_encoder = clip_model_frozen.visual

    def extract_features(self, image, classnames, modality='photo'):
        """Extract features using the specified modality branch."""
        # 1. Get Prompts from Learner
        learner = self.prompt_learner_photo if modality == 'photo' else self.prompt_learner_sketch
        (
            text_input, 
            tokenized_prompts, 
            first_visual_prompt, 
            deeper_text_prompts, 
            deeper_visual_prompts
        ) = learner(classnames)
        
        # 2. Extract Text Features
        text_features = self.text_encoder(text_input, tokenized_prompts, deeper_text_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 3. Extract Visual Features
        visual_features = self.visual_encoder(image, first_visual_prompt, deeper_visual_prompts)
        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        
        # 4. Extract Frozen Reference Features
        with torch.no_grad():
            frozen_features = self.frozen_visual_encoder(image.type(self.dtype))
            frozen_features_norm = frozen_features / frozen_features.norm(dim=-1, keepdim=True)
            
        # 5. Calculate Logits for Classification Loss
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * visual_features_norm @ text_features.t()
        
        return visual_features_norm, frozen_features_norm, text_features, logits

    def forward(self, x, classnames):
        """
        Forward pass for training.
        x: [photo_tensor, sk_tensor, photo_aug_tensor, sk_aug_tensor, neg_tensor, label]
        (Assuming input format matches CoPrompt_SBIR dataset)
        """
        photo_tensor, sk_tensor, photo_aug_tensor, sk_aug_tensor, neg_tensor, label = x
        
        # --- Photo Branch ---
        photo_feat, frozen_photo_feat, text_feat_photo, logits_photo = self.extract_features(photo_tensor, classnames, modality='photo')
        
        # --- Sketch Branch ---
        sketch_feat, frozen_sketch_feat, text_feat_sketch, logits_sketch = self.extract_features(sk_tensor, classnames, modality='sketch')
        
        # --- Negative Photo Branch (for Triplet Loss) ---
        neg_feat, _, _, _ = self.extract_features(neg_tensor, classnames, modality='photo')
        
        return (
            photo_feat, frozen_photo_feat, logits_photo,
            sketch_feat, frozen_sketch_feat, logits_sketch,
            neg_feat, label
        )


class HiCroPL_SBIR(pl.LightningModule):
    def __init__(self, cfg, args, classnames, model):
        super().__init__()
        self.cfg = cfg
        self.args = args
        self.classnames = classnames
        self.model = model
        
        # Temporary buffer for metrics
        self.test_photo_features = []
        self.test_sketch_features = []
        self.test_photo_labels = []
        self.test_sketch_labels = []

    def on_train_epoch_start(self):
        # Ensure that clip models are in correct mode during training (frozen aside from LN)
        self.model.clip_model_frozen.eval()
        self.model.clip_model.eval()
        self.model.clip_model.apply(set_ln_to_train)

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
        
        print(f"Number of trainable prompt params: {sum(p.numel() for p in prompt_params):,}")
        print(f"Number of trainable LayerNorm params: {sum(p.numel() for p in clip_ln_params):,}")
        
        prompt_lr = getattr(self.cfg, 'prompt_lr', 1e-5)
        clip_ln_lr = getattr(self.cfg, 'clip_LN_lr', 1e-5)
        weight_decay = getattr(self.cfg, 'weight_decay', 1e-4)
        
        optimizer = torch.optim.Adam([
            {'params': prompt_params, 'lr': prompt_lr},
            {'params': clip_ln_params, 'lr': clip_ln_lr}
        ], weight_decay=weight_decay)
        
        return optimizer

    def training_step(self, batch, batch_idx):
        from src.losses import loss_fn_hicropl
        
        # Unpack batch and push to device happens in Lightning automatically, but we might need label explicitly
        features = self.model(batch, self.classnames)
        
        # Calculate custom loss
        loss = loss_fn_hicropl(self.args, features)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def extract_eval_features(self, tensor, modality):
        """Extract normalized visual features for evaluation/inference."""
        learner = self.model.prompt_learner_photo if modality == 'photo' else self.model.prompt_learner_sketch
        
        with torch.no_grad():
            _, _, first_visual_prompt, _, deeper_visual_prompts = learner(self.classnames)
            visual_features = self.model.visual_encoder(tensor, first_visual_prompt, deeper_visual_prompts)
            visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
            
        return visual_features_norm

    def validation_step(self, batch, batch_idx):
        tensor, label, type_data = batch
        
        # Split batch based on modality type (Assuming type_data: 0 for sketch, 1 for photo)
        idx_sketch = (type_data == 0)
        idx_photo = (type_data == 1)
        
        if idx_sketch.any():
            sketch_feat = self.extract_eval_features(tensor[idx_sketch], modality='sketch')
            self.test_sketch_features.append(sketch_feat)
            self.test_sketch_labels.append(label[idx_sketch])
            
        if idx_photo.any():
            photo_feat = self.extract_eval_features(tensor[idx_photo], modality='photo')
            self.test_photo_features.append(photo_feat)
            self.test_photo_labels.append(label[idx_photo])

    def on_validation_epoch_end(self):
        if not self.test_photo_features or not self.test_sketch_features:
            print("Warning: Missing features for validation. Skipping metrics.")
            return

        gallery_features = torch.cat(self.test_photo_features, dim=0)
        gallery_labels = torch.cat(self.test_photo_labels, dim=0)
        
        query_features = torch.cat(self.test_sketch_features, dim=0)
        query_labels = torch.cat(self.test_sketch_labels, dim=0)
        
        # Set metrics according to dataset (following paper and model_LN_prompt logic)
        dataset = getattr(self.opts, 'dataset', 'sketchy') if hasattr(self, 'opts') else getattr(self.args, 'dataset', 'sketchy')
        if dataset == 'sketchy_ext':
            map_k = 200
            p_k = 200
        elif dataset == 'tuberlin':
            map_k = 0
            p_k = 100
        elif dataset == 'quickdraw':
            map_k = 0
            p_k = 200
        else:  # sketchy (basic)
            map_k = 0
            p_k = 200
            
        # Calculate Cosine Similarity Matrix
        sim_matrix = query_features @ gallery_features.T
        
        # Create boolean target matrix [n_queries, n_gallery]
        target_matrix = query_labels.unsqueeze(1) == gallery_labels.unsqueeze(0)
        
        val_map = 0.0
        # Calculate map
        if map_k != 0:
            top_k_actual = min(map_k, len(gallery_features))
            val_map = retrieval_average_precision(sim_matrix, target_matrix, top_k=top_k_actual)
            self.log(f'val_map_{map_k}', val_map, prog_bar=True, logger=True)
        else:
            val_map = retrieval_average_precision(sim_matrix, target_matrix)
            self.log('val_map_all', val_map, prog_bar=True, logger=True)
            
        val_p = retrieval_precision(sim_matrix, target_matrix, top_k=p_k)
        self.log(f'val_p_{p_k}', val_p, prog_bar=True, logger=True)
        
        if map_k != 0:
            print(f"\n[Validation] mAP@{map_k}: {val_map:.4f}, P@{p_k}: {val_p:.4f}")
        else:
            print(f"\n[Validation] mAP@all: {val_map:.4f}, P@{p_k}: {val_p:.4f}")

        # Clear buffers
        self.test_photo_features.clear()
        self.test_sketch_features.clear()
        self.test_photo_labels.clear()
        self.test_sketch_labels.clear()

    # Reuse validation logic for testing
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
