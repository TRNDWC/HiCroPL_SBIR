import copy
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision

from src.clip import clip as _clip
from src.clip.model import QuickGELU


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


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class AttentionPooling(nn.Module):
    """LKP (Layer-specific Knowledge Proxy): compress n_ctx prompt tokens into a 1-token proxy.

    A learnable proxy token attends (via MHA) to the prompt tokens of one layer. Per-layer
    instances are cloned outside this module (one per source layer & direction).
    Input shapes are unbatched: [L, E]. PyTorch MHA supports unbatched 2D queries since 1.9.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, token_query, sequence_key, sequence_value):
        token_query = token_query + self.attn(
            self.ln_1(token_query), self.ln_1(sequence_key), self.ln_1(sequence_value),
            need_weights=False,
        )[0]
        token_query = self.ln_2(token_query)
        return token_query


class CrossPromptAttention(nn.Module):
    """Knowledge Mapper: query prompts attend to encoded key/value via cross-attention + FFN.

    One instance per direction (photo->sketch or sketch->photo). Query and key/value can
    have different hidden sizes; here both are vis_dim=768 because both ends are visual.
    """

    def __init__(self, hidden_size: int, encoder_hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(encoder_hidden_size, hidden_size)
        self.linear_v = nn.Linear(encoder_hidden_size, hidden_size)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(hidden_size, hidden_size * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(hidden_size * 4, hidden_size)),
        ]))
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, q, k, v):
        q_proj = self.linear_q(q)
        k_proj = self.linear_k(k)
        v_proj = self.linear_v(v)
        q_proj = q_proj + self.attn(self.ln_1(q_proj), self.ln_1(k_proj), self.ln_1(v_proj), need_weights=False)[0]
        q_proj = q_proj + self.ffn(self.ln_2(q_proj))
        return q_proj


class CustomCLIP(nn.Module):
    """Cross-domain (XDom) hierarchical prompt learning, ported from HiCroPL.

    Shared CLIP visual encoder with TWO deep per-domain prompt sets (sketch, photo). Each
    domain has a ParameterList of L=prompt_depth tensors, shape (n_ctx, vis_dim). Layer 0
    prompts get concatenated to the patch stream before the transformer; layers 1..L-1
    prompts swap into the trailing n_ctx slots inside ResidualAttentionBlock_XDom.

    Cross-domain knowledge exchange (re-purposing HiCroPL's T<->I as Photo<->Sketch):
      - Layers i in [1, cross_layer): photo guides sketch. LKP(photo) compresses each
        photo[i] -> 1 proxy token; CrossPromptAttention(photo->sketch) updates sketch[i].
      - Layers i in [cross_layer, prompt_depth): sketch guides photo. Symmetric.
      - Layer 0 stays per-domain (anchor token of each domain).

    Text branch (per-modality CoOp + L_ce) is kept SHALLOW exactly as before.
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

        def _count_trainable(m):
            total = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable

        c_tot, c_tr = _count_trainable(self.clip)
        print(f"clip (visual + text, freeze_all_but_bn): trainable {c_tr:,} / total {c_tot:,}")

        self.logit_scale = self.clip.logit_scale

        # ZS-residual ensemble (CoPrompt/HiCroPL style): combine the deep-prompted feature with
        # a prompt-free pass through the SAME (LN+QKV-trainable) backbone, then renormalize.
        # A robust generalist stream (prompt-free) regularizes the prompt-adapted specialist.
        self.use_residual = bool(int(getattr(cfg, "zs_residual", 1)))

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

        # --- Cross-domain deep visual prompts (1 ParameterList per domain, one tensor per layer).
        prompt_dim_v = self.clip.visual.conv1.weight.shape[0]
        n_ctx = int(getattr(cfg, "n_ctx", 3))
        prompt_depth = int(getattr(cfg, "prompt_depth", 1))
        cross_layer = int(getattr(cfg, "cross_layer", prompt_depth // 2))
        n_heads = int(getattr(cfg, "mapper_heads", 8))
        assert prompt_depth >= 1, "prompt_depth must be >= 1"
        assert 0 <= cross_layer <= prompt_depth, "cross_layer must lie in [0, prompt_depth]"
        self.n_ctx = n_ctx
        self.prompt_depth = prompt_depth
        self.cross_layer = cross_layer

        def _new_prompt_list():
            return nn.ParameterList([
                nn.Parameter(torch.empty(n_ctx, prompt_dim_v, dtype=self.dtype))
                for _ in range(prompt_depth)
            ])

        self.cross_prompts_photo = _new_prompt_list()
        self.cross_prompts_sketch = _new_prompt_list()
        if n_ctx > 0:
            for p in self.cross_prompts_photo:
                nn.init.normal_(p, std=0.02)
            for p in self.cross_prompts_sketch:
                nn.init.normal_(p, std=0.02)

        # --- Knowledge Mappers (cross-attention + FFN), one per direction.
        # photo -> sketch: applies to layers i in [1, cross_layer). Active iff cross_layer > 1.
        # sketch -> photo: applies to layers i in [cross_layer, prompt_depth). Active iff prompt_depth > cross_layer.
        self.photo2sketch_net = CrossPromptAttention(
            hidden_size=prompt_dim_v, encoder_hidden_size=prompt_dim_v, num_attention_heads=n_heads,
        )
        self.sketch2photo_net = CrossPromptAttention(
            hidden_size=prompt_dim_v, encoder_hidden_size=prompt_dim_v, num_attention_heads=n_heads,
        )

        # --- LKP modules: per-layer AttentionPooling + a learnable proxy token.
        # Photo proxies are the sources for layers [1, cross_layer): n_photo_proxy = max(cross_layer - 1, 0).
        # Sketch proxies are the sources for layers [cross_layer, prompt_depth): n_sketch_proxy = prompt_depth - cross_layer.
        n_photo_proxy = max(cross_layer - 1, 0)
        n_sketch_proxy = max(prompt_depth - cross_layer, 0)
        self.attn_pool_photo_nets = _get_clones(
            AttentionPooling(prompt_dim_v, n_heads), n_photo_proxy,
        ) if n_photo_proxy > 0 else nn.ModuleList()
        self.attn_pool_sketch_nets = _get_clones(
            AttentionPooling(prompt_dim_v, n_heads), n_sketch_proxy,
        ) if n_sketch_proxy > 0 else nn.ModuleList()
        self.photo_proxy_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, prompt_dim_v, dtype=self.dtype) * 0.02)
            for _ in range(n_photo_proxy)
        ])
        self.sketch_proxy_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, prompt_dim_v, dtype=self.dtype) * 0.02)
            for _ in range(n_sketch_proxy)
        ])

        # --- Per-modality shallow text prompts (CoOp), unchanged from baseline.
        prompt_dim_t = self.clip.ln_final.weight.shape[0]
        self.text_prompt_sketch = nn.Parameter(torch.empty(n_ctx, prompt_dim_t, dtype=self.dtype))
        self.text_prompt_photo = nn.Parameter(torch.empty(n_ctx, prompt_dim_t, dtype=self.dtype))
        if n_ctx > 0:
            nn.init.normal_(self.text_prompt_sketch, std=0.02)
            nn.init.normal_(self.text_prompt_photo, std=0.02)

        with torch.no_grad():
            embed_photo = self.clip.token_embedding(self.tokenized_photo).type(self.dtype)
            embed_sketch = self.clip.token_embedding(self.tokenized_sketch).type(self.dtype)
        self.register_buffer("token_prefix_photo", embed_photo[:, :1, :])
        self.register_buffer("token_suffix_photo", embed_photo[:, 1 + n_ctx:, :])
        self.register_buffer("token_prefix_sketch", embed_sketch[:, :1, :])
        self.register_buffer("token_suffix_sketch", embed_sketch[:, 1 + n_ctx:, :])

    def compute_cross_prompts(self):
        """Run the photo<->sketch knowledge exchange and return per-layer prompts.

        Returns
        -------
        photo_layers : list[Tensor]   length = prompt_depth, each [n_ctx, vis_dim]
        sketch_layers : list[Tensor]  length = prompt_depth, each [n_ctx, vis_dim]
        """
        photo_layers = [p for p in self.cross_prompts_photo]
        sketch_layers = [p for p in self.cross_prompts_sketch]

        # ---- photo -> sketch on layers [1, cross_layer)
        if self.cross_layer > 1:
            proxies = []
            for j, i in enumerate(range(1, self.cross_layer)):
                proxies.append(self.attn_pool_photo_nets[j](
                    token_query=self.photo_proxy_tokens[j],
                    sequence_key=photo_layers[i],
                    sequence_value=photo_layers[i],
                ))
            kv = torch.cat(proxies, dim=0)                                 # [cross_layer-1, vis_dim]
            sk_stack = torch.stack(sketch_layers[1:self.cross_layer], dim=0)
            n_layers, n_ctx_, vis_dim_ = sk_stack.shape
            q = sk_stack.reshape(n_layers * n_ctx_, vis_dim_)              # [(cross_layer-1)*n_ctx, vis_dim]
            updated = self.photo2sketch_net(q, kv, kv)
            updated = updated.reshape(n_layers, n_ctx_, vis_dim_)
            for j, i in enumerate(range(1, self.cross_layer)):
                sketch_layers[i] = updated[j]

        # ---- sketch -> photo on layers [cross_layer, prompt_depth)
        if self.prompt_depth > self.cross_layer:
            proxies = []
            for j, i in enumerate(range(self.cross_layer, self.prompt_depth)):
                proxies.append(self.attn_pool_sketch_nets[j](
                    token_query=self.sketch_proxy_tokens[j],
                    sequence_key=sketch_layers[i],
                    sequence_value=sketch_layers[i],
                ))
            kv = torch.cat(proxies, dim=0)
            ph_stack = torch.stack(photo_layers[self.cross_layer:self.prompt_depth], dim=0)
            n_layers, n_ctx_, vis_dim_ = ph_stack.shape
            q = ph_stack.reshape(n_layers * n_ctx_, vis_dim_)
            updated = self.sketch2photo_net(q, kv, kv)
            updated = updated.reshape(n_layers, n_ctx_, vis_dim_)
            for j, i in enumerate(range(self.cross_layer, self.prompt_depth)):
                photo_layers[i] = updated[j]

        return photo_layers, sketch_layers

    def encode_visual(self, x, modality, photo_layers=None, sketch_layers=None):
        """Encode an image with deep, per-domain prompts.

        Convenience for callers that do not pre-compute the cross-domain prompts (e.g. eval).
        Training should call `compute_cross_prompts()` once and reuse the result across the
        three forward passes to avoid redundant cross-attention.
        """
        if photo_layers is None or sketch_layers is None:
            photo_layers, sketch_layers = self.compute_cross_prompts()

        layers = sketch_layers if modality == "sketch" else photo_layers
        x = x.type(self.dtype)
        if self.n_ctx == 0 or self.prompt_depth == 0:
            prompted = self.clip.encode_image(x)
        else:
            shallow = layers[0].unsqueeze(0).expand(x.shape[0], -1, -1)    # [B, n_ctx, vis_dim]
            deeper = layers[1:] if self.prompt_depth > 1 else None
            prompted = self.clip.encode_image(x, prompt=shallow, deeper_prompts=deeper)

        if not self.use_residual:
            return prompted

        # Prompt-free generalist stream through the same backbone, ensembled with the prompted one.
        plain = self.clip.encode_image(x)
        prompted = prompted / prompted.norm(dim=-1, keepdim=True)
        plain = plain / plain.norm(dim=-1, keepdim=True)
        mixed = prompted + plain
        return mixed / mixed.norm(dim=-1, keepdim=True)

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
        if ctx.numel() == 0:
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

        photo_layers, sketch_layers = self.compute_cross_prompts()
        sketch_feat = self.encode_visual(sk_tensor, "sketch", photo_layers, sketch_layers)
        photo_feat = self.encode_visual(photo_tensor, "photo", photo_layers, sketch_layers)
        neg_feat = self.encode_visual(neg_tensor, "photo", photo_layers, sketch_layers)

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
        m = self.model
        self.print(
            f"XDom prompt: depth={m.prompt_depth}, cross_layer={m.cross_layer}, n_ctx={m.n_ctx}; "
            f"photo prompts: {len(m.cross_prompts_photo)} layers, sketch prompts: {len(m.cross_prompts_sketch)} layers; "
            f"text shallow tokens: sketch={m.text_prompt_sketch.shape[0]}, photo={m.text_prompt_photo.shape[0]}"
        )
        try:
            self.log('prompt_depth', m.prompt_depth, prog_bar=False, logger=True)
            self.log('cross_layer', m.cross_layer, prog_bar=False, logger=True)
            self.log('tokens_text_sketch', m.text_prompt_sketch.shape[0], prog_bar=False, logger=True)
            self.log('tokens_text_photo', m.text_prompt_photo.shape[0], prog_bar=False, logger=True)
        except Exception:
            pass

    def configure_optimizers(self):
        """Three Adam groups.

        1. ``clip_params`` (LN + naked Parameters + MHA QKV after ``freeze_all_but_bn``) at ``clip_LN_lr``.
        2. ``prompt_params`` (deep visual ParameterLists + shallow text prompts) at ``prompt_lr``.
        3. ``mapper_lkp_params`` (CrossPromptAttention + AttentionPooling + proxy tokens) at ``mapper_lr``.
        """
        m = self.model
        prompt_params = []
        prompt_params.extend(list(m.cross_prompts_photo))
        prompt_params.extend(list(m.cross_prompts_sketch))
        if m.text_prompt_sketch.requires_grad:
            prompt_params.append(m.text_prompt_sketch)
        if m.text_prompt_photo.requires_grad:
            prompt_params.append(m.text_prompt_photo)

        mapper_lkp_params = []
        mapper_lkp_params.extend(p for p in m.photo2sketch_net.parameters() if p.requires_grad)
        mapper_lkp_params.extend(p for p in m.sketch2photo_net.parameters() if p.requires_grad)
        mapper_lkp_params.extend(p for p in m.attn_pool_photo_nets.parameters() if p.requires_grad)
        mapper_lkp_params.extend(p for p in m.attn_pool_sketch_nets.parameters() if p.requires_grad)
        mapper_lkp_params.extend(list(m.photo_proxy_tokens))
        mapper_lkp_params.extend(list(m.sketch_proxy_tokens))

        clip_params = list(m.clip.parameters())
        clip_trainable = sum(p.numel() for p in clip_params if p.requires_grad)

        self.print(f"Trainable prompt params (deep visual + shallow text): {sum(p.numel() for p in prompt_params):,}")
        self.print(f"Trainable mapper + LKP params: {sum(p.numel() for p in mapper_lkp_params):,}")
        self.print(f"Trainable clip params (LN + naked + MHA QKV): {clip_trainable:,}")

        prompt_lr = getattr(self.cfg, 'prompt_lr', 1e-5)
        mapper_lr = getattr(self.cfg, 'mapper_lr', prompt_lr)
        clip_ln_lr = getattr(self.cfg, 'clip_LN_lr', 1e-5)
        weight_decay = getattr(self.cfg, 'weight_decay', 1e-4)

        groups = [
            {'params': clip_params, 'lr': clip_ln_lr},
            {'params': prompt_params, 'lr': prompt_lr},
        ]
        if mapper_lkp_params:
            groups.append({'params': mapper_lkp_params, 'lr': mapper_lr})
        return torch.optim.Adam(groups, weight_decay=weight_decay)

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

        self.test_photo_features.clear()
        self.test_sketch_features.clear()
        self.test_photo_labels.clear()
        self.test_sketch_labels.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
