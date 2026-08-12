"""
HiCroPL - Hierarchical Cross-modal Prompt Learning components.
Adapted from https://github.com/zzeoZheng/HiCroPL for ZS-SBIR task.

Components:
    - AttentionPooling: Layer-specific Knowledge Proxy (LKP)
    - CrossPromptAttention: Multi-scale Knowledge Mapper
    - TextEncoder: CLIP text encoder with deep prompt injection
    - VisualEncoder: CLIP ViT encoder with deep prompt injection
    - CrossModalPromptLearner: Bidirectional knowledge flow (text <-> visual)
"""

import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class QuickGELU(nn.Module):
    """Fast GELU approximation (same as in CLIP)."""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def _get_clones(module, N):
    """Create N deep copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _kmeans(x, k, n_iters=25):
    """Minimal Lloyd's-algorithm k-means (pure torch, no new dependency).

    Used for data-driven prompt initialization (SPT / VIPAMIN-style): cluster
    real patch embeddings instead of drawing the prompt from pure Gaussian
    noise. x: [N, D]. Returns centroids [k, D].
    """
    n = x.shape[0]
    k = min(k, n)
    idx = torch.randperm(n, device=x.device)[:k]
    centroids = x[idx].clone()
    for _ in range(n_iters):
        dists = torch.cdist(x, centroids)  # [N, k]
        assign = dists.argmin(dim=1)  # [N]
        new_centroids = centroids.clone()
        for c in range(k):
            mask = assign == c
            if mask.any():
                new_centroids[c] = x[mask].mean(dim=0)
        centroids = new_centroids
    return centroids

class TextEncoder(nn.Module):
    # GIỮ NGUYÊN 100% TỪ BẢN GỐC HICROPL
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, cross_prompts_text_deeper):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        combined = [x, cross_prompts_text_deeper] # <-- cái này là cải tiến của HiCroPL
        outputs = self.transformer(combined) # <-- cái transformer coi như hộp đen của pretrained model
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class AttentionPooling(nn.Module):
    # GIỮ NGUYÊN 100% TỪ BẢN GỐC HICROPL
    def __init__(self, hidden_size, num_attention_heads):
        super(AttentionPooling, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, token_query, sequence_key, sequence_value):
        token_query = token_query + self.attn(self.ln_1(token_query), self.ln_1(sequence_key), self.ln_1(sequence_value), need_weights=False)[0]
        token_query = self.ln_2(token_query)
        return token_query


class CrossPromptAttention(nn.Module):
    # GIỮ NGUYÊN 100% TỪ BẢN GỐC HICROPL
    def __init__(self, hidden_size, encoder_hidden_size, num_attention_heads):
        super(CrossPromptAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.linear_q = nn.Linear(hidden_size, hidden_size) 
        self.linear_k = nn.Linear(encoder_hidden_size, hidden_size) 
        self.linear_v = nn.Linear(encoder_hidden_size, hidden_size) 
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(hidden_size, hidden_size * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(hidden_size * 4, hidden_size))
        ]))
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, q, k, v):
        q_proj = self.linear_q(q)
        k_proj = self.linear_k(k)
        v_proj = self.linear_v(v)
        q_proj = q_proj + self.attn(self.ln_1(q_proj), self.ln_1(k_proj), self.ln_1(v_proj), need_weights=False)[0]
        q_proj = q_proj + self.ffn(self.ln_2(q_proj))
        return q_proj


class CrossModalPromptLearner(nn.Module):
    """Bidirectional text<->visual prompt exchange for ONE branch (photo OR
    sketch) -- faithful port of the original HiCroPL T<->I mapping
    (github.com/zzeoZheng/HiCroPL/blob/main/trainers/hicropl.py), instantiated
    once per branch by CustomCLIP when `use_text_visual_exchange=True`
    (separate weights, no coupling between the two instances -- the
    photo<->sketch case is a different, mutually-exclusive architecture, see
    VisualVisualPromptLearner).

    Update rule matches VisualVisualPromptLearner's fix for the same bug:
    plain REPLACEMENT on a local Python list (`current_x[i] = updated[i]`),
    never `.data.copy_()` on the stored nn.Parameter (that breaks autograd).
    No gate/additive-residual -- kept consistent with
    VisualVisualPromptLearner's actual current behavior on this branch (no
    gate there either), so the two architectures differ ONLY in exchange
    topology (photo<->sketch vs text<->visual-per-branch), not in update rule.
    """

    def __init__(self, cfg, classnames, clip_model, sample_images=None):
        super().__init__()

        n_cls = len(classnames)
        self.prompt_depth = getattr(cfg, 'prompt_depth', 9)
        cross_layer = getattr(cfg, 'cross_layer', -1)
        self.cross_layer = self.prompt_depth // 2 if cross_layer < 0 else cross_layer
        n_ctx = getattr(cfg, 'n_ctx', 4)
        ctx_init = getattr(cfg, 'ctx_init', "a photo of a")
        prec = getattr(cfg, 'prec', "fp32")

        assert self.prompt_depth >= 1, "Language prompt depth should be >=1"
        assert 0 <= self.cross_layer <= self.prompt_depth, "cross_layer must be in [0, prompt_depth]"

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        v_dim = 768

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.dtype = dtype
        self.token_embedding = clip_model.token_embedding

        ######## cross-modal text token initialization ########
        if ctx_init and (n_ctx) <= 4:
            ctx_init_clean = ctx_init.replace("_", " ")
            from src.clip import clip as _clip
            prompt = _clip.tokenize(ctx_init_clean)
            text_device = clip_model.token_embedding.weight.device
            prompt = prompt.to(text_device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init_clean
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        cross_prompts_text = nn.ParameterList([self.ctx] + [nn.Parameter(torch.empty(n_ctx, 512, dtype=dtype)) for _ in range(self.prompt_depth - 1)])
        for single_para in cross_prompts_text[1:]:
            nn.init.normal_(single_para, std=0.02)
        self.cross_prompts_text = cross_prompts_text

        ######## cross-modal visual token initialization ########
        # Layer-0 ONLY data-driven init (SPT/VIPAMIN-style, same technique as
        # VisualVisualPromptLearner): k-means over this branch's own real
        # patch embeddings (conv1 output, frozen) when sample images are
        # given; deeper layers always stay plain Gaussian. Falls back to
        # plain Gaussian for layer 0 too when no sample images are provided.
        self.has_visual_anchor = sample_images is not None
        if self.has_visual_anchor:
            with torch.no_grad():
                sample_images = sample_images.to(
                    device=clip_model.visual.conv1.weight.device, dtype=dtype
                )
                patches = clip_model.visual.conv1(sample_images)  # [B, v_dim, grid, grid]
                patches = patches.reshape(patches.shape[0], patches.shape[1], -1)  # [B, v_dim, grid*grid]
                patches = patches.permute(0, 2, 1).reshape(-1, patches.shape[1])   # [B*grid*grid, v_dim]
                visual_centroids = _kmeans(patches.float(), n_ctx).to(dtype)
            visual_layer0 = visual_centroids.clone()
        else:
            visual_layer0 = torch.empty(n_ctx, v_dim, dtype=dtype)
            nn.init.normal_(visual_layer0, std=0.02)

        cross_prompts_visual = nn.ParameterList(
            [nn.Parameter(visual_layer0)] +
            [nn.Parameter(torch.empty(n_ctx, v_dim, dtype=dtype)) for _ in range(self.prompt_depth - 1)]
        )
        for single_para in cross_prompts_visual[1:]:
            nn.init.normal_(single_para, std=0.02)
        self.cross_prompts_visual = cross_prompts_visual

        ######## knowledge mapper network and LKP ########
        self.text2visual_net = CrossPromptAttention(hidden_size=v_dim, encoder_hidden_size=ctx_dim, num_attention_heads=8)
        self.visual2text_net = CrossPromptAttention(hidden_size=ctx_dim, encoder_hidden_size=v_dim, num_attention_heads=8)
        if prec == "fp16":
            self.text2visual_net, self.visual2text_net = self.text2visual_net.half(), self.visual2text_net.half()

        attn_pooling_text = AttentionPooling(hidden_size=ctx_dim, num_attention_heads=8)
        self.attn_pooling_text_nets = _get_clones(attn_pooling_text, self.cross_layer)
        attn_pooling_visual = AttentionPooling(hidden_size=v_dim, num_attention_heads=8)
        self.attn_pooling_visual_nets = _get_clones(attn_pooling_visual, self.prompt_depth - self.cross_layer)

        text_proxy_token = torch.randn(1, ctx_dim, dtype=dtype)
        self.text_proxy_tokens = nn.ParameterList([nn.Parameter(text_proxy_token) for _ in range(self.cross_layer)])
        visual_proxy_token = torch.randn(1, v_dim, dtype=dtype)
        self.visual_proxy_tokens = nn.ParameterList([nn.Parameter(visual_proxy_token) for _ in range(self.cross_layer, self.prompt_depth)])

        if prec == "fp16":
            self.attn_pooling_text_nets, self.attn_pooling_visual_nets = self.attn_pooling_text_nets.half(), self.attn_pooling_visual_nets.half()

        ######## Initialize prompts for all classes ########
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        from src.clip import clip as _clip
        tokenized_prompts = torch.cat([_clip.tokenize(p) for p in prompts]).to(clip_model.token_embedding.weight.device)
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # Register as buffers so they move with the model
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        # Local mutable copies -- entries get REPLACED here, never the stored
        # nn.Parameter itself, so gradients flow correctly into the mapper/LKP
        # networks (fix for the original `.data.copy_()` bug; matches
        # VisualVisualPromptLearner's plain-replacement style -- no gate).
        current_text_prompts = list(self.cross_prompts_text)
        current_visual_prompts = list(self.cross_prompts_visual)

        ctx = current_text_prompts[0]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # Construct text input prompts
        text_input = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        ######## T->I mapping (shallow layers [0, cross_layer)) ########
        if self.cross_layer > 0:
            proxy_text_tokens = []
            for i in range(self.cross_layer):
                text_proxy_token = self.attn_pooling_text_nets[i](
                    token_query=self.text_proxy_tokens[i],
                    sequence_key=current_text_prompts[i],
                    sequence_value=current_text_prompts[i]
                )
                proxy_text_tokens.append(text_proxy_token)
            proxy_text_prompts = torch.cat(proxy_text_tokens, dim=0)

            visual_prompts_range = torch.cat(
                [current_visual_prompts[i].unsqueeze(0) for i in range(self.cross_layer)], dim=0
            )
            visual_prompts_flat = visual_prompts_range.view(-1, visual_prompts_range.shape[-1])
            proxy_text_flat = proxy_text_prompts.view(-1, proxy_text_prompts.shape[-1])

            updated_visual_prompts = self.text2visual_net(visual_prompts_flat, proxy_text_flat, proxy_text_flat)
            updated_visual_prompts = updated_visual_prompts.view(
                self.cross_layer, -1, updated_visual_prompts.shape[-1]
            )
            for i in range(self.cross_layer):
                current_visual_prompts[i] = updated_visual_prompts[i]
        ######## T->I end ########

        ######## I->T mapping (deep layers [cross_layer, prompt_depth)) ########
        n_deep = self.prompt_depth - self.cross_layer
        if n_deep > 0:
            proxy_visual_tokens = []
            for i in range(self.cross_layer, self.prompt_depth):
                visual_proxy_token = self.attn_pooling_visual_nets[i - self.cross_layer](
                    token_query=self.visual_proxy_tokens[i - self.cross_layer],
                    sequence_key=current_visual_prompts[i],
                    sequence_value=current_visual_prompts[i]
                )
                proxy_visual_tokens.append(visual_proxy_token)
            proxy_visual_prompts = torch.cat(proxy_visual_tokens, dim=0)

            text_prompts_range = torch.cat(
                [current_text_prompts[i].unsqueeze(0) for i in range(self.cross_layer, self.prompt_depth)], dim=0
            )
            text_prompts_flat = text_prompts_range.view(-1, text_prompts_range.shape[-1])
            proxy_visual_flat = proxy_visual_prompts.view(-1, proxy_visual_prompts.shape[-1])

            updated_text_prompts = self.visual2text_net(text_prompts_flat, proxy_visual_flat, proxy_visual_flat)
            updated_text_prompts = updated_text_prompts.view(n_deep, -1, updated_text_prompts.shape[-1])
            for i in range(self.cross_layer, self.prompt_depth):
                current_text_prompts[i] = updated_text_prompts[i - self.cross_layer]
        ######## I->T end ########

        cross_prompts_text_deeper = [current_text_prompts[i] for i in range(1, len(current_text_prompts))]
        cross_prompts_visual_deeper = [current_visual_prompts[i] for i in range(1, len(current_visual_prompts))]

        return text_input, current_visual_prompts[0], cross_prompts_text_deeper, cross_prompts_visual_deeper


class VisualVisualPromptLearner(nn.Module):
    """Bidirectional cross-domain prompt exchange between photo and sketch.

    Faithful port of the original HiCroPL CrossModalPromptLearner's T<->I
    mapping mechanics (github.com/zzeoZheng/HiCroPL/blob/main/trainers/hicropl.py),
    adapted from text<->visual to photo<->sketch:
      - Photo plays the "text" role (source, shallow layers [0, cross_layer)):
        LKP compresses each shallow-layer photo prompt into one proxy token;
        all proxies in the range are concatenated and fed through ONE joint
        CrossPromptAttention (Mapper) call that produces the new sketch
        prompts for that same range.
      - Sketch plays the "visual" role (source, deep layers
        [cross_layer, prompt_depth)): symmetric direction, updates photo.

    Update rule matches the ORIGINAL HiCroPL exactly: plain REPLACEMENT, no
    gate, no additive residual -- `current_x_prompts[i] = updated[i]` on a
    local Python list, never `.data.copy_()` on the stored nn.Parameter (that
    breaks autograd; see CrossModalPromptLearner above, which has that bug
    from a faulty port and is otherwise unused/dead code).

    Note: this joint-per-range Mapper call (not a per-layer causal proxy
    accumulation) is what the actual original repo does, which differs from
    the "Option 1 refactored" design note's pseudocode -- going with the real
    original mechanics here per explicit request.

    Ablation: `disable_exchange=True` (cfg.disable_exchange) skips both
    mapping blocks in forward() while leaving everything else in this class
    (k-means layer-0 init, mapper/LKP construction) untouched --
    cross_prompts_photo/sketch then train as fully independent per-branch
    prompts, isolating the exchange itself as the only variable between a
    paired ON/OFF comparison.
    """

    def __init__(self, cfg, clip_model_photo, clip_model_sketch, sample_photo_images=None, sample_sketch_images=None):
        super().__init__()

        self.prompt_depth = getattr(cfg, 'prompt_depth', 9)
        n_ctx = getattr(cfg, 'n_ctx', 4)
        cross_layer = getattr(cfg, 'cross_layer', -1)
        self.cross_layer = self.prompt_depth // 2 if cross_layer < 0 else cross_layer
        # Clean on/off switch for the exchange itself, for ablation against
        # the exact same codebase (same k-means init, same prompt_depth,
        # same LR) -- everything below still gets constructed identically
        # either way; only forward()
        # skips the two mapping blocks when disabled, so
        # cross_prompts_photo/sketch just stay as their own independently
        # -trained values (matches ducta/baseline's no-exchange design).
        self.disable_exchange = getattr(cfg, 'disable_exchange', False)

        assert self.prompt_depth >= 1
        assert 0 <= self.cross_layer <= self.prompt_depth, "cross_layer must be in [0, prompt_depth]"

        dtype = clip_model_photo.dtype
        p_dim = clip_model_photo.visual.conv1.weight.shape[0]   # 768
        s_dim = clip_model_sketch.visual.conv1.weight.shape[0]  # 768
        assert p_dim == s_dim, "Both branches must have same embedding dimension"

        self.dtype = dtype
        self.n_ctx = n_ctx

        ######## photo prompt initialization (base, per layer) ########
        # Data-driven init for layer 0 ONLY (SPT/VIPAMIN-style): k-means over
        # real photo patch embeddings (conv1 output, frozen, in-distribution
        # for CLIP's own pretraining). Deeper layers (1..depth-1) stay plain
        # Gaussian.
        #
        # Reverted from a broader "reuse centroids for every layer" variant
        # (both photo and sketch) after checking MaPLe's own ablation (Table
        # 8, arXiv:2210.03117): "informed init at ALL layers" scored WORSE
        # (HM 77.88) than plain random at all layers (HM 78.52), and best was
        # informed init at layer 0 only, random elsewhere (HM 78.55) -- the
        # opposite of what extending our k-means init to every layer does.
        # DA-VPT (arXiv:2505.23694, Sec 5.3.2) independently found the same
        # pattern: per-layer mean/data-derived init "impedes effectiveness"
        # due to homogeneous content across layers hurting discriminative
        # learning. Both ablate on base->novel-class generalization, which is
        # the same axis ZS-SBIR evaluates on (unseen categories), so the
        # caution transfers directly. Layer-0-only grounding matches both
        # papers' best config and is also what our own A/B/C/D ablation
        # actually validated (mAP 0.7799); the all-layers variant's +0.0024
        # was 1-2 epoch noise, not a confirmed gain.
        #
        # One-time init value only -- no persistent regularization loss pulls
        # the prompt back toward this anchor during training (removed; loss
        # was gradient-negligible at any practical weight, see git history).
        # Gradient is free to move ctx_photo/ctx_sketch away from this
        # starting point. Sketch gets the same layer-0-only k-means init as
        # photo below (own conv1).
        self.has_photo_anchor = sample_photo_images is not None
        if self.has_photo_anchor:
            with torch.no_grad():
                sample_photo_images = sample_photo_images.to(
                    device=clip_model_photo.visual.conv1.weight.device, dtype=dtype
                )
                patches = clip_model_photo.visual.conv1(sample_photo_images)  # [B, p_dim, grid, grid]
                patches = patches.reshape(patches.shape[0], patches.shape[1], -1)  # [B, p_dim, grid*grid]
                patches = patches.permute(0, 2, 1).reshape(-1, patches.shape[1])   # [B*grid*grid, p_dim]
                photo_centroids = _kmeans(patches.float(), n_ctx).to(dtype)
            photo_vectors = photo_centroids.clone()
        else:
            photo_vectors = torch.empty(n_ctx, p_dim, dtype=dtype)
            nn.init.normal_(photo_vectors, std=0.02)

        self.ctx_photo = nn.Parameter(photo_vectors)
        cross_prompts_photo = nn.ParameterList(
            [self.ctx_photo] +
            [nn.Parameter(torch.empty(n_ctx, p_dim, dtype=dtype))
             for _ in range(self.prompt_depth - 1)]
        )
        for single_para in cross_prompts_photo[1:]:
            nn.init.normal_(single_para, std=0.02)
        self.cross_prompts_photo = cross_prompts_photo
        ######## photo prompt initialization end ########

        ######## sketch prompt initialization (base, per layer) ########
        # Same technique as photo, layer-0 only: k-means over sketch's own
        # conv1 patches. Layers 1..depth-1 stay plain Gaussian, matching the
        # photo branch exactly (MaPLe/DA-VPT: only ground layer 0, see note
        # above).
        self.has_sketch_anchor = sample_sketch_images is not None
        if self.has_sketch_anchor:
            with torch.no_grad():
                sample_sketch_images = sample_sketch_images.to(
                    device=clip_model_sketch.visual.conv1.weight.device, dtype=dtype
                )
                s_patches = clip_model_sketch.visual.conv1(sample_sketch_images)  # [B, s_dim, grid, grid]
                s_patches = s_patches.reshape(s_patches.shape[0], s_patches.shape[1], -1)
                s_patches = s_patches.permute(0, 2, 1).reshape(-1, s_patches.shape[1])  # [B*grid*grid, s_dim]
                sketch_centroids = _kmeans(s_patches.float(), n_ctx).to(dtype)
            sketch_vectors = sketch_centroids.clone()
        else:
            sketch_vectors = torch.empty(n_ctx, s_dim, dtype=dtype)
            nn.init.normal_(sketch_vectors, std=0.02)

        self.ctx_sketch = nn.Parameter(sketch_vectors)
        cross_prompts_sketch = nn.ParameterList(
            [self.ctx_sketch] +
            [nn.Parameter(torch.empty(n_ctx, s_dim, dtype=dtype))
             for _ in range(self.prompt_depth - 1)]
        )
        for single_para in cross_prompts_sketch[1:]:
            nn.init.normal_(single_para, std=0.02)
        self.cross_prompts_sketch = cross_prompts_sketch
        ######## sketch prompt initialization end ########

        ######## Knowledge mapper networks (orig: text2visual_net / visual2text_net) ########
        # LKP (attn_pooling_*) is a SINGLE instance shared across every layer
        # in its direction -- NOT `_get_clones`-ed per layer. Per the original
        # LKP design, the layer-specific part is the proxy TOKEN (a distinct
        # learnable query per layer, `photo_proxy_token[i]`/`sketch_proxy_token[i]`),
        # not the pooling network itself; the network is the same
        # cross-attention operator applied with a different query each layer.
        # Cuts trainable params from ~46.2M to ~22.5M (LKP was 61% of the
        # total at cross_layer=6/prompt_depth=12, entirely from per-layer
        # cloning -- see param-decomposition audit).
        if self.cross_layer > 0:
            self.photo2sketch_net = CrossPromptAttention(hidden_size=s_dim, encoder_hidden_size=p_dim, num_attention_heads=8)

            self.attn_pooling_photo_net = AttentionPooling(hidden_size=p_dim, num_attention_heads=8)

            photo_proxy_token = torch.randn(1, p_dim, dtype=dtype)
            self.photo_proxy_token = nn.ParameterList(
                [nn.Parameter(photo_proxy_token.clone()) for _ in range(self.cross_layer)]
            )

        n_deep = self.prompt_depth - self.cross_layer
        if n_deep > 0:
            self.sketch2photo_net = CrossPromptAttention(hidden_size=p_dim, encoder_hidden_size=s_dim, num_attention_heads=8)

            self.attn_pooling_sketch_net = AttentionPooling(hidden_size=s_dim, num_attention_heads=8)

            sketch_proxy_token = torch.randn(1, s_dim, dtype=dtype)
            self.sketch_proxy_token = nn.ParameterList(
                [nn.Parameter(sketch_proxy_token.clone()) for _ in range(self.cross_layer, self.prompt_depth)]
            )
        ######## Knowledge mapper end ########

    def forward(self):
        # Local mutable copies -- entries get REPLACED here, never the stored
        # nn.Parameter itself, so gradients flow correctly into the mapper/LKP
        # networks (matches the original HiCroPL exactly; see class docstring).
        current_photo_prompts = list(self.cross_prompts_photo)
        current_sketch_prompts = list(self.cross_prompts_sketch)

        ######## Photo -> Sketch mapping (shallow layers [0, cross_layer)) ########
        if not self.disable_exchange and self.cross_layer > 0:
            proxy_photo_tokens = []
            for i in range(self.cross_layer):
                photo_proxy_token = self.attn_pooling_photo_net(
                    token_query=self.photo_proxy_token[i],
                    sequence_key=current_photo_prompts[i],
                    sequence_value=current_photo_prompts[i],
                )
                proxy_photo_tokens.append(photo_proxy_token)
            proxy_photo_prompts = torch.cat(proxy_photo_tokens, dim=0)

            sketch_prompts_range = torch.cat(
                [current_sketch_prompts[i].unsqueeze(0) for i in range(self.cross_layer)], dim=0
            )
            sketch_prompts_flat = sketch_prompts_range.view(-1, sketch_prompts_range.shape[-1])
            proxy_photo_flat = proxy_photo_prompts.view(-1, proxy_photo_prompts.shape[-1])

            updated_sketch_prompts = self.photo2sketch_net(sketch_prompts_flat, proxy_photo_flat, proxy_photo_flat)
            updated_sketch_prompts = updated_sketch_prompts.view(
                self.cross_layer, -1, updated_sketch_prompts.shape[-1]
            )
            for i in range(self.cross_layer):
                current_sketch_prompts[i] = updated_sketch_prompts[i]
        ######## Photo -> Sketch end ########

        ######## Sketch -> Photo mapping (deep layers [cross_layer, prompt_depth)) ########
        n_deep = self.prompt_depth - self.cross_layer
        if not self.disable_exchange and n_deep > 0:
            proxy_sketch_tokens = []
            for i in range(self.cross_layer, self.prompt_depth):
                sketch_proxy_token = self.attn_pooling_sketch_net(
                    token_query=self.sketch_proxy_token[i - self.cross_layer],
                    sequence_key=current_sketch_prompts[i],
                    sequence_value=current_sketch_prompts[i],
                )
                proxy_sketch_tokens.append(sketch_proxy_token)
            proxy_sketch_prompts = torch.cat(proxy_sketch_tokens, dim=0)

            photo_prompts_range = torch.cat(
                [current_photo_prompts[i].unsqueeze(0) for i in range(self.cross_layer, self.prompt_depth)], dim=0
            )
            photo_prompts_flat = photo_prompts_range.view(-1, photo_prompts_range.shape[-1])
            proxy_sketch_flat = proxy_sketch_prompts.view(-1, proxy_sketch_prompts.shape[-1])

            updated_photo_prompts = self.sketch2photo_net(photo_prompts_flat, proxy_sketch_flat, proxy_sketch_flat)
            updated_photo_prompts = updated_photo_prompts.view(
                n_deep, -1, updated_photo_prompts.shape[-1]
            )
            for i in range(self.cross_layer, self.prompt_depth):
                current_photo_prompts[i] = updated_photo_prompts[i - self.cross_layer]
        ######## Sketch -> Photo end ########

        cross_prompts_photo_deeper = [current_photo_prompts[i] for i in range(1, len(current_photo_prompts))]
        cross_prompts_sketch_deeper = [current_sketch_prompts[i] for i in range(1, len(current_sketch_prompts))]

        return (
            current_photo_prompts[0],
            current_sketch_prompts[0],
            cross_prompts_photo_deeper,
            cross_prompts_sketch_deeper,
        )


class SimpleTextPromptLearner(nn.Module):
    """Minimal text-only prompt learner: prepares tokenized prompts and text prompt tensors.

    Matches the outputs needed by TextEncoder but does not perform cross-modal mapping.
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        self.prompt_depth = getattr(cfg, 'prompt_depth', 9)
        n_ctx = getattr(cfg, 'n_ctx', 4)
        ctx_init = getattr(cfg, 'ctx_init', "a photo of a")
        dtype = clip_model.dtype

        ctx_dim = clip_model.ln_final.weight.shape[0]

        # initialize context vectors
        if ctx_init and (n_ctx) <= 4:
            from src.clip import clip as _clip
            prompt = _clip.tokenize(ctx_init.replace("_", " "))
            prompt = prompt.to(clip_model.token_embedding.weight.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init.replace("_", " ")
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        cross_prompts_text = nn.ParameterList([self.ctx] + [nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=dtype)) for _ in range(self.prompt_depth - 1)])
        for p in cross_prompts_text[1:]:
            nn.init.normal_(p, std=0.02)
        self.cross_prompts_text = cross_prompts_text

        # register token prefix/suffix buffers for class prompts
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        from src.clip import clip as _clip
        tokenized_prompts = torch.cat([_clip.tokenize(p) for p in prompts]).to(clip_model.token_embedding.weight.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
            ctx = ctx[label]  # Select ctx by label to match batch dimension
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self, label=None):
        ctx = self.cross_prompts_text[0]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.tokenized_prompts.shape[0], -1, -1)
        text_input = self.construct_prompts(ctx, self.token_prefix, self.token_suffix, label=label)
        cross_prompts_text_deeper = [self.cross_prompts_text[i] for i in range(1, len(self.cross_prompts_text))]
        return text_input, cross_prompts_text_deeper


class VisualEncoder(nn.Module):
    """Wraps CLIP VisionTransformer_HiCroPL for deep prompt injection.
    
    Delegates directly to VisionTransformer_HiCroPL.forward(x, img_prompts,
    cross_prompts_visual_deeper).
    """

    def __init__(self, clip_model):
        super().__init__()
        self.vit = clip_model.visual  # VisionTransformer_HiCroPL
        self.dtype = clip_model.dtype

    def forward(self, image, first_visual_prompt, deeper_visual_prompts):
        """
        Args:
            image: [B, 3, 224, 224]
            first_visual_prompt: [n_ctx, v_dim] - shallow prompt for layer 0
            deeper_visual_prompts: list of L-1 tensors [n_ctx, v_dim] for layers 1..L-1
        Returns:
            [B, embed_dim] - image features
        """
        return self.vit(image.type(self.dtype), first_visual_prompt, deeper_visual_prompts)

    