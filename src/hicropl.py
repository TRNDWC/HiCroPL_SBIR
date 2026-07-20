"""
Simple deep prompt learning for ZS-SBIR: one learnable token set per transformer
layer, no cross-modal exchange. Depth is independent per branch:
    --vision_depth  -> number of ViT layers (photo AND sketch) with a prompt
    --text_depth    -> number of text transformer layers (photo AND sketch) with a prompt

Components:
    - DeepPromptLearner: generic per-layer learnable token container
    - VisualPromptLearner: wraps DeepPromptLearner for one visual branch
    - TextPromptLearner: wraps DeepPromptLearner for one text branch + per-class prompts
    - TextEncoder / VisualEncoder: thin wrappers feeding prompts into CLIP's
      per-layer injection mechanism (ResidualAttentionBlock_HiCroPL)
"""

import torch
import torch.nn as nn


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


class ContentCrossAttention(nn.Module):
    """Cross-attend one layer's static prompt tokens (query) onto a per-sample
    content descriptor (single-token key/value), producing a per-sample
    modulated prompt. Unlike self/pooling attention over a static bank, the
    key/value sequence length is 1 (the descriptor), so this is effectively a
    per-sample FiLM-style modulation of the query's own tokens.
    """

    def __init__(self, prompt_dim, descriptor_dim, num_heads=8):
        super().__init__()
        self.linear_k = nn.Linear(descriptor_dim, prompt_dim)
        self.linear_v = nn.Linear(descriptor_dim, prompt_dim)
        self.attn = nn.MultiheadAttention(embed_dim=prompt_dim, num_heads=num_heads)
        self.ln_q = nn.LayerNorm(prompt_dim)
        self.ln_out = nn.LayerNorm(prompt_dim)

    def forward(self, query, descriptor):
        """
        query: (n_ctx, dim) -- this layer's own static prompt tokens
        descriptor: (B, descriptor_dim) -- per-sample content descriptor
        returns: (B, n_ctx, dim) -- per-sample modulated prompt
        """
        B = descriptor.shape[0]
        q = self.ln_q(query).unsqueeze(1).expand(-1, B, -1)  # (n_ctx, B, dim)
        k = self.linear_k(descriptor).unsqueeze(0)  # (1, B, dim)
        v = self.linear_v(descriptor).unsqueeze(0)  # (1, B, dim)
        out, _ = self.attn(q, k, v, need_weights=False)  # (n_ctx, B, dim)
        out = self.ln_out(out)
        return out.permute(1, 0, 2)  # (B, n_ctx, dim)


class VisualPromptLearnerConditioned(nn.Module):
    """Deep prompt learner for the sketch branch: the usual static per-layer
    tokens (inherited from VisualPromptLearner), plus an optional per-sample
    modulation driven by a photo content descriptor (H2 hypothesis).

    forward(descriptor=None):
        descriptor is None   -> (shallow, deeper), same 2-tuple as
                                 VisualPromptLearner (static only, used at
                                 eval time and under modality dropout, since a
                                 query sketch has no paired photo at retrieval
                                 time).
        descriptor (B, D)    -> (shallow, deeper, gate_values), where shallow
                                 is (B, n_ctx, dim) and each deeper layer is
                                 (B, n_ctx, dim) -- per-sample blend of the
                                 static prompt and a cross-attention modulation
                                 conditioned on descriptor. CLIP's own prompt
                                 injection (VisionTransformer_HiCroPL.forward,
                                 ResidualAttentionBlock_HiCroPL.forward) already
                                 handles a per-sample (B, n_ctx, dim) prompt
                                 correctly via a no-op expand() when the batch
                                 dim already matches, so no changes are needed
                                 there.
    """

    def __init__(self, cfg, clip_model):
        super().__init__()
        self.base = VisualPromptLearner(cfg, clip_model)
        depth = getattr(cfg, 'vision_depth', 1)
        prompt_dim = clip_model.visual.conv1.weight.shape[0]
        descriptor_dim = clip_model.visual.proj.shape[1]
        self.cross_attn = nn.ModuleList([
            ContentCrossAttention(prompt_dim, descriptor_dim) for _ in range(depth)
        ])
        self.gate_logit = nn.Parameter(torch.zeros(depth, dtype=clip_model.dtype))

    def forward(self, descriptor=None):
        shallow, deeper = self.base()
        if descriptor is None:
            return shallow, deeper

        layers = [shallow] + deeper
        gates = torch.sigmoid(self.gate_logit)
        blended = []
        for layer, cross_attn, gate in zip(layers, self.cross_attn, gates):
            own = layer.unsqueeze(0).expand(descriptor.shape[0], -1, -1)  # (B, n_ctx, dim)
            mod = cross_attn(layer, descriptor)  # (B, n_ctx, dim)
            # ContentCrossAttention ends in LayerNorm (~unit scale), but prompt
            # tokens are initialized at std=0.02 to match CLIP's expected
            # activation range -> rescale mod to own's per-token norm so gate
            # actually controls the blend ratio instead of being swamped by a
            # ~50x scale mismatch (own is (n_ctx,1), mod is (B,n_ctx,1) -- broadcasts).
            own_norm = layer.norm(dim=-1, keepdim=True)  # (n_ctx, 1)
            mod_norm = mod.norm(dim=-1, keepdim=True)  # (B, n_ctx, 1)
            mod = mod * (own_norm / (mod_norm + 1e-6))
            blended.append((1 - gate) * own + gate * mod)

        gate_values = [g.item() for g in gates]
        return blended[0], blended[1:], gate_values


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
