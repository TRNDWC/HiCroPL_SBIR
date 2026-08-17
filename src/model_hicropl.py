import copy
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision

from src.hicropl import (
    TextEncoder,
    VisualEncoder,
    VisualVisualPromptLearner,
    SimpleTextPromptLearner,
    CrossModalPromptLearner,
)


def freeze_model(m):
    """Freeze all parameters of the given module."""
    for param in m.parameters():
        param.requires_grad_(False)


def freeze_all_but_bn(model):
    """Freeze every parameter except those owned by nn.LayerNorm modules.

    Matches by module membership (not attribute name) so it correctly covers
    parameters that aren't literally named `weight`/`bias`, e.g.
    nn.MultiheadAttention's `in_proj_weight`/`in_proj_bias`, or loose
    nn.Parameters like `class_embedding`/`positional_embedding`/`proj`/
    `text_projection`/`logit_scale` — all of which must stay frozen per the
    CLIP-AT design (only LayerNorm trainable; Attention and MLP frozen).
    """
    ln_param_ids = {
        id(p)
        for m in model.modules() if isinstance(m, torch.nn.LayerNorm)
        for p in m.parameters()
    }
    for p in model.parameters():
        if id(p) not in ln_param_ids:
            p.requires_grad_(False)


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


class CustomCLIP(nn.Module):
    """
    HiCroPL-SBIR Architecture Wrapper.
    Sử dụng HiCroPLFeatureExtractor làm nòng cốt.
    """

    def __init__(self, cfg, clip_model, classnames=None, sample_photo_images=None, sample_sketch_images=None):
        super().__init__()
        self.cfg = cfg
        # Ablation: no visual/text prompt learning at all -- only LayerNorm
        # trainable (matches ducta/baseline's CLIP-AT recipe). Requires
        # clip_model to already be a vanilla (non-prompted) build --
        # experiments/hicropl_prompt.py forces clip_trainer='CoOp' before
        # load_clip_to_cpu when this flag is set, so `self.clip.encode_image`/
        # `encode_text` work standalone (no prompt tensors required).
        self.no_prompt_learning = getattr(cfg, 'no_prompt_learning', False)
        # Alternative architecture: per-branch text<->visual exchange (see
        # CrossModalPromptLearner) instead of the default photo<->sketch
        # VisualVisualPromptLearner. Mutually exclusive with the default path
        # -- --disable_exchange has NO effect here (it only gates the
        # photo<->sketch mapping blocks inside VisualVisualPromptLearner,
        # which isn't constructed at all when this is set).
        self.use_text_visual_exchange = getattr(cfg, 'use_text_visual_exchange', False)

        if classnames is None:
            classnames = []
        if len(classnames) == 0:
            raise ValueError("CustomCLIP requires non-empty classnames during initialization.")

        original_device = next(clip_model.parameters()).device
        self.dtype = clip_model.dtype

        # 1. Single shared backbone for both photo and sketch (matches ducta/baseline:
        # one CLIP copy, same LayerNorm weights updated by gradients from both modalities).
        self.clip = copy.deepcopy(clip_model).to(original_device)
        freeze_all_but_bn(self.clip)

        # Print trainable param counts for verification
        total = sum(p.numel() for p in self.clip.parameters())
        trainable = sum(p.numel() for p in self.clip.parameters() if p.requires_grad)
        print(f"clip (shared): trainable {trainable:,} / total {total:,} params")

        # Single shared logit scale (matches ducta/baseline)
        self.logit_scale = self.clip.logit_scale

        if self.no_prompt_learning:
            print("[ABLATION] no_prompt_learning=True: skipping ALL prompt learners. "
                  "Only LayerNorm is trainable; text uses the fixed ctx_init/ctx_init_sketch template.")
            from src.clip import clip as _clip
            classnames_clean = [name.replace("_", " ") for name in classnames]
            ctx_init_photo = getattr(cfg, 'ctx_init', 'a photo of a').replace("_", " ")
            ctx_init_sketch = getattr(cfg, 'ctx_init_sketch', 'a sketch of a').replace("_", " ")
            prompts_photo = [f"{ctx_init_photo} {name}." for name in classnames_clean]
            prompts_sketch = [f"{ctx_init_sketch} {name}." for name in classnames_clean]
            # Fixed (non-learnable) tokenized templates -- registered as buffers,
            # not nn.Parameter, so they never appear in configure_optimizers.
            self.register_buffer(
                "tokenized_prompts_photo",
                torch.cat([_clip.tokenize(p) for p in prompts_photo]).to(original_device),
            )
            self.register_buffer(
                "tokenized_prompts_sketch",
                torch.cat([_clip.tokenize(p) for p in prompts_sketch]).to(original_device),
            )
        elif self.use_text_visual_exchange:
            # Per-branch bidirectional text<->visual exchange -- ONLY text and
            # visual of the SAME domain ever interact. Two fully independent
            # instances (separate weights, no shared modules, no coupling):
            # text_visual_learner_photo only ever sees photo text + photo
            # visual; text_visual_learner_sketch only ever sees sketch text +
            # sketch visual. Neither instance references the other, and
            # visual_visual_learner/text_prompt_photo/text_prompt_sketch are
            # NOT constructed in this branch at all.
            print("Initializing Photo Text<->Visual Exchange Learner...")
            cfg_photo = copy.copy(cfg)
            cfg_photo.ctx_init = getattr(cfg, 'ctx_init', 'a photo of a')
            self.text_visual_learner_photo = CrossModalPromptLearner(
                cfg_photo, classnames, self.clip, sample_images=sample_photo_images
            )

            print("Initializing Sketch Text<->Visual Exchange Learner...")
            cfg_sketch = copy.copy(cfg)
            cfg_sketch.ctx_init = getattr(cfg, 'ctx_init_sketch', 'a sketch of a')
            self.text_visual_learner_sketch = CrossModalPromptLearner(
                cfg_sketch, classnames, self.clip, sample_images=sample_sketch_images
            )

            self.text_encoder_photo = TextEncoder(self.clip)
            self.text_encoder_sketch = TextEncoder(self.clip)
            self.visual_encoder_photo = VisualEncoder(self.clip)
            self.visual_encoder_sketch = VisualEncoder(self.clip)
        else:
            # -- Prompt Learners --
            # Initialize Visual-Visual learner + simple text learners + adapters
            print("Initializing Visual Prompt Learner (photo + sketch, independent)...")
            self.visual_visual_learner = VisualVisualPromptLearner(
                cfg, self.clip, self.clip,
                sample_photo_images=sample_photo_images, sample_sketch_images=sample_sketch_images
            )

            print("Initializing Photo Text Prompt Learner...")
            cfg_photo = copy.copy(cfg)
            cfg_photo.ctx_init = getattr(cfg, 'ctx_init', 'a photo of a')
            self.text_prompt_photo = SimpleTextPromptLearner(cfg_photo, classnames, self.clip)

            print("Initializing Sketch Text Prompt Learner...")
            cfg_sketch = copy.copy(cfg)
            cfg_sketch.ctx_init = getattr(cfg, 'ctx_init_sketch', 'a sketch of a')
            self.text_prompt_sketch = SimpleTextPromptLearner(cfg_sketch, classnames, self.clip)

            # -- Encoders (both branches wrap the SAME shared backbone) --
            self.text_encoder_photo = TextEncoder(self.clip)
            self.text_encoder_sketch = TextEncoder(self.clip)
            self.visual_encoder_photo = VisualEncoder(self.clip)
            self.visual_encoder_sketch = VisualEncoder(self.clip)

        # -- Attribute-guided auxiliary loss (ArGue-inspired, see report) --
        # A (attr_emb) and N (neg_emb) are frozen, pre-computed, class-name
        # aligned tensors loaded once here as non-persistent buffers (not
        # nn.Parameter -- never trainable, never added to any optimizer
        # param group, and .detach()'d again at the point of use in
        # losses_hicropl.py as a second, redundant safety net per project
        # constraint #3). No new learnable parameter is added to the photo
        # branch or anywhere else by this block.
        self.use_attr_loss = getattr(cfg, 'use_attr_loss', False)
        if self.use_attr_loss:
            import json as _json
            attr_file = getattr(cfg, 'attr_file', 'data/attr_emb.npy')
            neg_file = getattr(cfg, 'neg_file', 'data/neg_emb.npy')
            attr_random = getattr(cfg, 'attr_random', False)

            attr_json_path = os.path.join(os.path.dirname(attr_file), 'attributes_seen.json') \
                if os.path.dirname(attr_file) else 'data/attributes_seen.json'
            if not os.path.exists(attr_json_path):
                raise FileNotFoundError(
                    f"--use_attr_loss requires '{attr_json_path}' (produced by "
                    f"tools/gen_attributes.py) to verify class-index alignment between "
                    f"attr_emb.npy's rows and classnames -- file not found."
                )
            with open(attr_json_path) as f:
                attr_json = _json.load(f)
            attr_classnames = sorted(attr_json['attributes'].keys())
            if list(classnames) != attr_classnames:
                raise RuntimeError(
                    f"--use_attr_loss: classnames passed to CustomCLIP "
                    f"(n={len(classnames)}) do not exactly match the class list/order in "
                    f"'{attr_json_path}' (n={len(attr_classnames)}) -- attr_emb.npy's rows "
                    f"would silently misalign with logits_attr's columns (wrong CE target "
                    f"per class) if allowed to proceed. First mismatch: "
                    f"{[c for c in classnames if c not in attr_classnames][:5]} not in attr "
                    f"file / {[c for c in attr_classnames if c not in list(classnames)][:5]} "
                    f"not in classnames."
                )

            attr_np = np.load(attr_file)  # (104, 8, 512), L2-normalized, see tools/gen_attributes.py
            neg_np = np.load(neg_file)    # (32, 512), L2-normalized, see tools/gen_neg_bank.py
            attr_tensor = torch.from_numpy(attr_np).to(dtype=self.dtype)
            neg_tensor = torch.from_numpy(neg_np).to(dtype=self.dtype)

            if attr_random:
                # Fixed, DEDICATED generator -- does not consume/perturb the
                # global RNG stream used by the rest of training (constraint:
                # "không đổi seed"). Same shape as the real attr_emb, also
                # L2-normalized for a fair, scale-matched ablation.
                gen = torch.Generator().manual_seed(20240101)
                attr_tensor = torch.randn(attr_tensor.shape, generator=gen, dtype=self.dtype)
                attr_tensor = attr_tensor / attr_tensor.norm(dim=-1, keepdim=True)
                print("[ABLATION] --attr_random: A replaced with a fixed-seed random tensor "
                      "of identical shape (content ablation, see report Run R4).")

            self.register_buffer("attr_emb", attr_tensor, persistent=False)
            self.register_buffer("neg_emb", neg_tensor, persistent=False)
            print(f"[CONFIG] --use_attr_loss: loaded attr_emb {tuple(attr_tensor.shape)} from "
                  f"'{attr_file}' (attr_random={attr_random}), neg_emb {tuple(neg_tensor.shape)} "
                  f"from '{neg_file}'.")

    def normalize_features(self, feat_prenorm):
        """L2-normalize feature tensors."""
        return feat_prenorm / feat_prenorm.norm(dim=-1, keepdim=True)

    def forward(self, x, classnames):
        """
        Forward pass for training with optimized redundancy.
        Calls visual learner ONCE and routes prompts by branch.
        """
        if len(x) == 5:
            sk_tensor, photo_tensor, neg_tensor, label, _filename = x
        else:
            sk_tensor, photo_tensor, neg_tensor, label = x[:4]

        if self.no_prompt_learning:
            # Plain frozen CLIP forward (only LayerNorm trainable) -- no
            # prompt tensors of any kind, text uses the fixed template.
            image_features_photo = self.clip.encode_image(photo_tensor.type(self.dtype))
            image_features_sketch = self.clip.encode_image(sk_tensor.type(self.dtype))
            image_features_neg = self.clip.encode_image(neg_tensor.type(self.dtype))
            text_features_all_photo = self.clip.encode_text(self.tokenized_prompts_photo)
            text_features_all_sketch = self.clip.encode_text(self.tokenized_prompts_sketch)
        elif self.use_text_visual_exchange:
            # Each branch's learner performs its OWN bidirectional text<->visual
            # exchange -- no coupling between the two learners/branches.
            text_input_photo_all, vis_shallow_photo, cross_prompts_text_deeper_photo, vis_deeper_photo = self.text_visual_learner_photo()
            text_features_all_photo = self.text_encoder_photo(text_input_photo_all, self.text_visual_learner_photo.tokenized_prompts, cross_prompts_text_deeper_photo)
            image_features_photo = self.visual_encoder_photo(photo_tensor.type(self.dtype), vis_shallow_photo, vis_deeper_photo)

            text_input_sketch_all, vis_shallow_sketch, cross_prompts_text_deeper_sketch, vis_deeper_sketch = self.text_visual_learner_sketch()
            text_features_all_sketch = self.text_encoder_sketch(text_input_sketch_all, self.text_visual_learner_sketch.tokenized_prompts, cross_prompts_text_deeper_sketch)
            image_features_sketch = self.visual_encoder_sketch(sk_tensor.type(self.dtype), vis_shallow_sketch, vis_deeper_sketch)

            # Negative branch (uses photo encoder + photo visual prompts)
            image_features_neg = self.visual_encoder_photo(neg_tensor.type(self.dtype), vis_shallow_photo, vis_deeper_photo)
        else:
            # 1. Call visual-visual learner ONCE (shared by both branches)
            photo_shallow, sketch_shallow, photo_deeper, sketch_deeper = self.visual_visual_learner()

            # 2. Photo branch: text learner + visual routing
            # Compute text features for ALL classes (not just batch) - needed for loss computation
            text_input_photo_all, cross_prompts_text_deeper_photo = self.text_prompt_photo(label=None)  # All classes
            text_features_all_photo = self.text_encoder_photo(text_input_photo_all, self.text_prompt_photo.tokenized_prompts, cross_prompts_text_deeper_photo)
            image_features_photo = self.visual_encoder_photo(photo_tensor.type(self.dtype), photo_shallow, photo_deeper)

            # 3. Sketch branch: text learner + visual routing
            # Compute text features for ALL classes (not just batch) - needed for loss computation
            text_input_sketch_all, cross_prompts_text_deeper_sketch = self.text_prompt_sketch(label=None)  # All classes
            text_features_all_sketch = self.text_encoder_sketch(text_input_sketch_all, self.text_prompt_sketch.tokenized_prompts, cross_prompts_text_deeper_sketch)
            image_features_sketch = self.visual_encoder_sketch(sk_tensor.type(self.dtype), sketch_shallow, sketch_deeper)

            # 4. Negative branch (uses photo encoder + photo visual prompts)
            image_features_neg = self.visual_encoder_photo(neg_tensor.type(self.dtype), photo_shallow, photo_deeper)

        # 5. Normalize features
        photo_feat = image_features_photo / image_features_photo.norm(dim=-1, keepdim=True)
        sketch_feat = image_features_sketch / image_features_sketch.norm(dim=-1, keepdim=True)
        neg_feat = image_features_neg / image_features_neg.norm(dim=-1, keepdim=True)
        text_feat_photo = text_features_all_photo / text_features_all_photo.norm(dim=-1, keepdim=True)
        text_feat_sketch = text_features_all_sketch / text_features_all_sketch.norm(dim=-1, keepdim=True)

        # 6. Compute logits
        logit_scale = self.logit_scale.exp()
        logits_photo = logit_scale * photo_feat @ text_feat_photo.t()
        logits_sketch = logit_scale * sketch_feat @ text_feat_sketch.t()

        return (
            photo_feat, logits_photo,
            sketch_feat, logits_sketch,
            neg_feat, label,
            text_feat_photo, text_feat_sketch,
            logit_scale,
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
        # NOTE: Encoders stay in training mode (required for LayerNorm to use batch statistics)
        # Setting eval() here would conflict with forward() expectation and break BN/LN behavior
        pass

    def on_fit_start(self):
        """Log the number of learnable prompt tokens per branch once at fit start.

        Logs four scalars (tokens count):
        - `tokens_visual_photo`
        - `tokens_visual_sketch`
        - `tokens_text_photo`
        - `tokens_text_sketch`
        """
        if self.model.use_text_visual_exchange:
            try:
                lp = self.model.text_visual_learner_photo
                ls = self.model.text_visual_learner_sketch
                tokens_visual_photo = len(lp.cross_prompts_visual) * lp.n_ctx
                tokens_visual_sketch = len(ls.cross_prompts_visual) * ls.n_ctx
                tokens_text_photo = len(lp.cross_prompts_text) * lp.n_ctx
                tokens_text_sketch = len(ls.cross_prompts_text) * ls.n_ctx
            except Exception:
                tokens_visual_photo = tokens_visual_sketch = tokens_text_photo = tokens_text_sketch = 0
        else:
            try:
                vv = self.model.visual_visual_learner
                # visual tokens: number of prompt vectors (prompt_depth * n_ctx)
                tokens_visual_photo = len(vv.cross_prompts_photo) * vv.n_ctx
                tokens_visual_sketch = len(vv.cross_prompts_sketch) * vv.n_ctx
            except Exception:
                tokens_visual_photo = 0
                tokens_visual_sketch = 0

            try:
                tp = self.model.text_prompt_photo
                ts = self.model.text_prompt_sketch
                tokens_text_photo = len(tp.cross_prompts_text) * tp.cross_prompts_text[0].shape[0]
                tokens_text_sketch = len(ts.cross_prompts_text) * ts.cross_prompts_text[0].shape[0]
            except Exception:
                tokens_text_photo = 0
                tokens_text_sketch = 0

        # Log to Lightning logger and print for immediate visibility
        self.print(f"Learnable tokens - visual/photo: {tokens_visual_photo}, visual/sketch: {tokens_visual_sketch}, text/photo: {tokens_text_photo}, text/sketch: {tokens_text_sketch}")
        # Use self.log so TensorBoard/other loggers capture these scalars
        # Use rank_zero_only to avoid duplicate logs in distributed runs
        try:
            self.log('tokens_visual_photo', tokens_visual_photo, prog_bar=True, logger=True)
            self.log('tokens_visual_sketch', tokens_visual_sketch, prog_bar=True, logger=True)
            self.log('tokens_text_photo', tokens_text_photo, prog_bar=True, logger=True)
            self.log('tokens_text_sketch', tokens_text_sketch, prog_bar=True, logger=True)
        except Exception:
            # Fallback to print-only if logger not ready
            pass

    def configure_optimizers(self):
        def add_unique_params(candidates, out_list, seen_ids):
            for p in candidates:
                if p.requires_grad and id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    out_list.append(p)

        seen_ids = set()

        prompt_params = []
        # Collect from whichever learner set is active (mutually exclusive).
        # These include all their internal params (CrossPromptAttention, AttentionPooling, etc.)
        # No learner submodules exist when no_prompt_learning=True (only
        # LayerNorm is trainable in that mode).
        if self.model.no_prompt_learning:
            learner_modules = set()
        elif self.model.use_text_visual_exchange:
            add_unique_params(self.model.text_visual_learner_photo.parameters(), prompt_params, seen_ids)
            add_unique_params(self.model.text_visual_learner_sketch.parameters(), prompt_params, seen_ids)
            learner_modules = {'text_visual_learner_photo', 'text_visual_learner_sketch'}
        else:
            add_unique_params(self.model.visual_visual_learner.parameters(), prompt_params, seen_ids)
            add_unique_params(self.model.text_prompt_photo.parameters(), prompt_params, seen_ids)
            add_unique_params(self.model.text_prompt_sketch.parameters(), prompt_params, seen_ids)
            learner_modules = {'visual_visual_learner', 'text_prompt_photo', 'text_prompt_sketch'}

        ln_params = []
        # Only collect LayerNorms from clip encoders (NOT from learners, already included above)
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                # Skip if inside a learner module (already included with learner params)
                if not any(learner_name in name for learner_name in learner_modules):
                    add_unique_params(module.parameters(recurse=False), ln_params, seen_ids)

        extra_trainable_params = []
        for _, p in self.model.named_parameters():
            if p.requires_grad and id(p) not in seen_ids:
                seen_ids.add(id(p))
                extra_trainable_params.append(p)

        non_prompt_params = ln_params + extra_trainable_params

        self.print(f"Number of trainable prompt params: {sum(p.numel() for p in prompt_params):,}")
        self.print(f"Number of trainable non-prompt params: {sum(p.numel() for p in non_prompt_params):,}")

        prompt_lr = getattr(self.cfg, 'prompt_lr', 1e-5)
        clip_ln_lr = getattr(self.cfg, 'clip_LN_lr', 1e-5)

        param_groups = []
        if prompt_params:
            param_groups.append({'params': prompt_params, 'lr': prompt_lr})
        if non_prompt_params:
            param_groups.append({'params': non_prompt_params, 'lr': clip_ln_lr})

        # No weight_decay (matches ducta/baseline's Adam call, which also omits it -> default 0).
        return torch.optim.Adam(param_groups)

    def on_after_backward(self):
        """Diagnostic: does gradient actually reach ctx_photo (layer-0 photo
        prompt)? If grad_norm prints ~0.000000 despite training, this tells us
        whether that's because the parameter never moves (grad ~0 -- real
        graph-disconnection bug) or because it does move but the angular
        drift relative to its own norm is just small at this lr/step count
        (grad non-zero, expected -- not a bug).

        No-op when no_prompt_learning=True -- ctx_photo doesn't exist in that
        mode (no prompts at all). When use_text_visual_exchange=True, checks
        the photo branch's own layer-0 text ctx instead (its closest analogue
        -- visual_visual_learner.ctx_photo doesn't exist in that mode either).
        """
        if self.model.no_prompt_learning:
            return
        if self.model.use_text_visual_exchange:
            ctx_photo = self.model.text_visual_learner_photo.ctx
        else:
            ctx_photo = self.model.visual_visual_learner.ctx_photo
        grad_norm = ctx_photo.grad.norm().item() if ctx_photo.grad is not None else 0.0
        param_norm = ctx_photo.detach().norm().item()
        self.log('ctx_photo_grad_norm', grad_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('ctx_photo_param_norm', param_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)

    def training_step(self, batch, batch_idx):
        from src.losses_hicropl import loss_fn_hicropl
        features = self.model(batch, self.classnames)
        loss = loss_fn_hicropl(self.args, features, model=self.model)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)

        return loss

    def extract_eval_features(self, tensor, modality):
        """Extract visual features (prompted only, no distill mixing)."""
        if self.model.no_prompt_learning:
            feat = self.model.clip.encode_image(tensor.type(self.model.dtype))
            return feat / feat.norm(dim=-1, keepdim=True)

        if self.model.use_text_visual_exchange:
            learner = (
                self.model.text_visual_learner_photo if modality == 'photo'
                else self.model.text_visual_learner_sketch
            )
            visual_encoder = (
                self.model.visual_encoder_photo if modality == 'photo'
                else self.model.visual_encoder_sketch
            )
            _, vis_shallow, _, vis_deeper = learner()
            feat = visual_encoder(tensor.type(self.model.dtype), vis_shallow, vis_deeper)
            return feat / feat.norm(dim=-1, keepdim=True)

        # Call visual learner once, cache outputs
        photo_shallow, sketch_shallow, photo_deeper, sketch_deeper = self.model.visual_visual_learner()

        if modality == 'photo':
            visual_encoder = self.model.visual_encoder_photo
            vis_shallow, vis_deeper = photo_shallow, photo_deeper
        else:
            visual_encoder = self.model.visual_encoder_sketch
            vis_shallow, vis_deeper = sketch_shallow, sketch_deeper

        feat = visual_encoder(tensor.type(self.model.dtype), vis_shallow, vis_deeper)
        return feat / feat.norm(dim=-1, keepdim=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._validation_step_category(batch, batch_idx, dataloader_idx)

    def _validation_step_category(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 3:
            tensor, label, type_data = batch
        else:
            tensor, label = batch
            type_data = None
            
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
        query_features   = torch.cat(self.test_sketch_features, dim=0).to(self.device)
        
        all_photo_category  = torch.cat(self.test_photo_labels, dim=0).to(self.device)
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

        grad_norm = self.trainer.callback_metrics.get("ctx_photo_grad_norm", None)
        param_norm = self.trainer.callback_metrics.get("ctx_photo_param_norm", None)
        if grad_norm is not None and param_norm is not None:
            self.print(f"[DEBUG] ctx_photo grad_norm (epoch avg): {grad_norm.item():.8f}, param_norm: {param_norm.item():.6f}")

        self.test_photo_features.clear()
        self.test_sketch_features.clear()
        self.test_photo_labels.clear()
        self.test_sketch_labels.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
