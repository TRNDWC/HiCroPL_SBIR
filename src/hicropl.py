"""
Deep prompt learning for ZS-SBIR.

Text branch (photo/sketch): independent per-layer learnable tokens, no exchange.
    --text_depth -> number of text transformer layers (photo AND sketch) with a prompt

Visual branch (photo/sketch): reuses HiCroPL's original exchange machinery
(AttentionPooling + CrossPromptAttention, https://github.com/zzeoZheng/HiCroPL)
verbatim, applied photo<->sketch instead of text<->image, with a fixed
directional split (no gate for now):
    [0, cross_layer)      : sketch -> photo
    [cross_layer, depth)  : photo -> sketch
Each direction compresses the source branch's per-layer prompt into a single
proxy token (AttentionPooling), then cross-attends the target branch's own
prompt onto that proxy (CrossPromptAttention) to produce the target's new
prompt for that layer -- a full replacement, matching the original HiCroPL
design (no learned blend gate yet). Unlike the original HiCroPL code (which
used an in-place `.data.copy_()` to swap prompt values -- that detaches from
autograd, so the mapper networks never receive gradient), this version keeps
the whole thing on the normal autograd graph via plain reassignment, so
sketch2photo_net/photo2sketch_net/the AttentionPooling nets are actually
trained by the main loss.

Components:
    - DeepPromptLearner: generic per-layer learnable token container (text only)
    - TextPromptLearner: wraps DeepPromptLearner for one text branch + per-class prompts
    - VisualPromptLearner: independent per-branch visual prompts (used when
      --use_visual_exchange is off)
    - AttentionPooling: compress a layer's prompt sequence into one proxy token
    - CrossPromptAttention: cross-attend target branch's prompt onto source's proxy
    - VisualVisualPromptLearner: owns both visual branches' prompts + the exchange
    - TextEncoder / VisualEncoder: thin wrappers feeding prompts into CLIP's
      per-layer injection mechanism (ResidualAttentionBlock_HiCroPL)
"""

from collections import OrderedDict

import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def _get_clones(module, n):
    import copy
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class DeepPromptLearner(nn.Module):
    """`depth` layers of (n_ctx, dim) learnable prompt tokens.

    Layer 0 ("shallow") is injected at the input, before the transformer.
    Layers 1..depth-1 ("deeper") each replace the previous layer's prompt
    tokens inside the corresponding transformer resblock.
    """

    def __init__(self, depth, n_ctx, dim, dtype, init_vectors=None):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"

        if init_vectors is not None:
            first = nn.Parameter(init_vectors.clone().type(dtype))
        else:
            first = nn.Parameter(torch.empty(n_ctx, dim, dtype=dtype))
            nn.init.normal_(first, std=0.02)

        layers = [first]
        for _ in range(depth - 1):
            layer = nn.Parameter(torch.empty(n_ctx, dim, dtype=dtype))
            nn.init.normal_(layer, std=0.02)
            layers.append(layer)

        self.prompts = nn.ParameterList(layers)

    def forward(self):
        shallow = self.prompts[0]
        deeper = [self.prompts[i] for i in range(1, len(self.prompts))]
        return shallow, deeper


class VisualPromptLearner(nn.Module):
    """Deep prompt learner for one visual branch (photo or sketch)."""

    def __init__(self, cfg, clip_model):
        super().__init__()
        depth = getattr(cfg, 'vision_depth', 1)
        n_ctx = getattr(cfg, 'n_ctx', 4)
        dim = clip_model.visual.conv1.weight.shape[0]
        dtype = clip_model.dtype
        self.learner = DeepPromptLearner(depth, n_ctx, dim, dtype)

    def forward(self):
        return self.learner()


class AttentionPooling(nn.Module):
    """Compress a sequence of prompt tokens into a single proxy token.
    Reused verbatim from HiCroPL's LKP (learnable key pooling)."""

    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, token_query, sequence_key, sequence_value):
        token_query = token_query + self.attn(
            self.ln_1(token_query), self.ln_1(sequence_key), self.ln_1(sequence_value), need_weights=False
        )[0]
        token_query = self.ln_2(token_query)
        return token_query


class CrossPromptAttention(nn.Module):
    """Cross-attend a target branch's own prompt (query) onto a source
    branch's proxy token (key/value), producing the target's new prompt for
    that layer. Reused verbatim from HiCroPL's knowledge mapper network."""

    def __init__(self, hidden_size, encoder_hidden_size, num_attention_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        # hidden_size is Q's dim, encoder_hidden_size is K/V's dim
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


class VisualVisualPromptLearner(nn.Module):
    """Per-layer learnable tokens for both visual branches (photo, sketch),
    with a directional cross-domain exchange split at `cross_layer`:
        [0, cross_layer)      : sketch -> photo
        [cross_layer, depth)  : photo -> sketch
    No gate yet -- full replacement of the target's prompt with the mapped
    value at each layer in its zone (matches original HiCroPL; a learned
    blend gate can be added later once this directional split is validated).
    """

    def __init__(self, cfg, clip_model_photo, clip_model_sketch):
        super().__init__()
        self.depth = getattr(cfg, 'vision_depth', 1)
        self.cross_layer = getattr(cfg, 'cross_layer', max(1, self.depth // 2))
        n_ctx = getattr(cfg, 'n_ctx', 4)

        assert 0 < self.cross_layer < self.depth, (
            f"cross_layer ({self.cross_layer}) must be strictly between 0 and "
            f"vision_depth ({self.depth}) so both zones (sketch->photo, photo->sketch) are non-empty."
        )

        dtype = clip_model_photo.dtype
        p_dim = clip_model_photo.visual.conv1.weight.shape[0]
        s_dim = clip_model_sketch.visual.conv1.weight.shape[0]
        assert p_dim == s_dim, "Both branches must share the same embedding dim"

        self.dtype = dtype
        self.n_ctx = n_ctx

        def _init_layers(depth, dim):
            layers = []
            for _ in range(depth):
                v = nn.Parameter(torch.empty(n_ctx, dim, dtype=dtype))
                nn.init.normal_(v, std=0.02)
                layers.append(v)
            return nn.ParameterList(layers)

        self.own_prompts_photo = _init_layers(self.depth, p_dim)
        self.own_prompts_sketch = _init_layers(self.depth, s_dim)

        # Direction 1: sketch -> photo, zone [0, cross_layer)
        self.sketch2photo_net = CrossPromptAttention(hidden_size=p_dim, encoder_hidden_size=s_dim, num_attention_heads=8)
        self.attn_pool_sketch_shallow = _get_clones(AttentionPooling(s_dim, 8), self.cross_layer)
        self.proxy_sketch_shallow = nn.ParameterList([
            nn.Parameter(torch.randn(1, s_dim, dtype=dtype) * 0.02) for _ in range(self.cross_layer)
        ])

        # Direction 2: photo -> sketch, zone [cross_layer, depth)
        self.photo2sketch_net = CrossPromptAttention(hidden_size=s_dim, encoder_hidden_size=p_dim, num_attention_heads=8)
        self.attn_pool_photo_deep = _get_clones(AttentionPooling(p_dim, 8), self.depth - self.cross_layer)
        self.proxy_photo_deep = nn.ParameterList([
            nn.Parameter(torch.randn(1, p_dim, dtype=dtype) * 0.02) for _ in range(self.depth - self.cross_layer)
        ])

    def forward(self):
        photo_prompts = list(self.own_prompts_photo)
        sketch_prompts = list(self.own_prompts_sketch)

        # --- sketch -> photo: zone [0, cross_layer) ---
        for i in range(self.cross_layer):
            proxy_sketch = self.attn_pool_sketch_shallow[i](
                token_query=self.proxy_sketch_shallow[i],
                sequence_key=self.own_prompts_sketch[i],
                sequence_value=self.own_prompts_sketch[i],
            )
            photo_prompts[i] = self.sketch2photo_net(self.own_prompts_photo[i], proxy_sketch, proxy_sketch)

        # --- photo -> sketch: zone [cross_layer, depth) ---
        for i in range(self.cross_layer, self.depth):
            j = i - self.cross_layer
            proxy_photo = self.attn_pool_photo_deep[j](
                token_query=self.proxy_photo_deep[j],
                sequence_key=self.own_prompts_photo[i],
                sequence_value=self.own_prompts_photo[i],
            )
            sketch_prompts[i] = self.photo2sketch_net(self.own_prompts_sketch[i], proxy_photo, proxy_photo)

        photo_shallow, photo_deeper = photo_prompts[0], photo_prompts[1:]
        sketch_shallow, sketch_deeper = sketch_prompts[0], sketch_prompts[1:]
        return photo_shallow, sketch_shallow, photo_deeper, sketch_deeper


class TextPromptLearner(nn.Module):
    """Deep prompt learner for one text branch (photo or sketch), plus the
    fixed prefix/suffix token embeddings needed to build a prompt per class.
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        depth = getattr(cfg, 'text_depth', 1)
        n_ctx = getattr(cfg, 'n_ctx', 4)
        ctx_init = getattr(cfg, 'ctx_init', 'a photo of a')
        dtype = clip_model.dtype
        dim = clip_model.ln_final.weight.shape[0]

        from src.clip import clip as _clip

        init_vectors = None
        prompt_prefix = " ".join(["X"] * n_ctx)
        if ctx_init and n_ctx <= 4:
            tokenized_init = _clip.tokenize(ctx_init.replace("_", " "))
            tokenized_init = tokenized_init.to(clip_model.token_embedding.weight.device)
            with torch.no_grad():
                init_embedding = clip_model.token_embedding(tokenized_init).type(dtype)
            init_vectors = init_embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init.replace("_", " ")

        self.learner = DeepPromptLearner(depth, n_ctx, dim, dtype, init_vectors=init_vectors)
        self.n_ctx = n_ctx

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([_clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.to(clip_model.token_embedding.weight.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def forward(self):
        ctx, deeper = self.learner()
        ctx = ctx.unsqueeze(0).expand(self.tokenized_prompts.shape[0], -1, -1)
        text_input = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return text_input, deeper


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, deeper_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.transformer([x, deeper_prompts])
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (highest token id in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class VisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.vit = clip_model.visual
        self.dtype = clip_model.dtype

    def forward(self, image, shallow_prompt, deeper_prompts):
        return self.vit(image.type(self.dtype), shallow_prompt, deeper_prompts)
