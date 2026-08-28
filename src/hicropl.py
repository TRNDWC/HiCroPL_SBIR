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


def _make_proxy_tokens(proxy_init, n_proxy, dim, dtype, source_layers):
    """Build one nn.Parameter [n_proxy, dim] per entry in source_layers -- the
    LKP query tokens (photo_proxy_token / sketch_proxy_token).

    'randn' replicates the original code path exactly: ONE torch.randn(1, dim)
    draw, cloned across layers (not one independent draw per layer) -- so
    --proxy_init randn --n_proxy 1 (the default) is bit-for-bit identical to
    the pre-ablation code under the same seed. 'small' and 'mean' draw/derive
    independently per layer, matching how every other std=0.02 tensor in this
    file is initialized (see cross_prompts_photo[1:]/cross_prompts_sketch[1:]).
    """
    num_layers = len(source_layers)
    if proxy_init == 'randn':
        base = torch.randn(n_proxy, dim, dtype=dtype)
        return nn.ParameterList([nn.Parameter(base.clone()) for _ in range(num_layers)])
    tokens = []
    for src in source_layers:
        if proxy_init == 'small':
            t = torch.empty(n_proxy, dim, dtype=dtype)
            nn.init.normal_(t, std=0.02)
        elif proxy_init == 'mean':
            mean_vec = src.mean(dim=0, keepdim=True).detach()  # [1, dim], this layer's own cross_prompts
            t = mean_vec.expand(n_proxy, dim).clone()  # independent storage, all n_proxy rows equal at init
        else:
            raise ValueError(f"Unknown --proxy_init: {proxy_init!r}")
        tokens.append(nn.Parameter(t))
    return nn.ParameterList(tokens)

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
        # Branch A: cut the gradient path from the sketch-side loss back into
        # ctx_photo/attn_pooling_photo through the Mapper, without changing
        # any numeric value (P~_photo.detach() as k/v). Isolates whether that
        # gradient feedback matters for the photo branch.
        self.exchange_detach_source = getattr(cfg, 'exchange_detach_source', False)
        # Branch B: replace P~_photo itself (the k/v fed to photo2sketch_net)
        # with an independent learned nn.Parameter unrelated to photo -- see
        # self.free_source below. Isolates whether photo-derived content is
        # what matters, or any learnable source into the Mapper suffices.
        self.exchange_free_source = getattr(cfg, 'exchange_free_source', False)
        # Control experiment: same pipeline as --exchange_detach_source
        # (same LKP module attn_pooling_photo_nets[i], same Mapper, same
        # detach before the Mapper) but the tensor fed into the LKP is
        # cross_prompts_sketch[i] instead of cross_prompts_photo[i] -- no
        # photo tensor participates in the Photo->Sketch block at all.
        # Isolates whether the block's benefit comes from the source being
        # PHOTO specifically, vs. any same-shape (detached) source flowing
        # through this exact pipeline. Adds/removes no parameters relative
        # to --exchange_detach_source.
        self.exchange_self_source = getattr(cfg, 'exchange_self_source', False)
        assert sum([self.exchange_detach_source, self.exchange_free_source, self.exchange_self_source]) <= 1, \
            "--exchange_detach_source, --exchange_free_source, and --exchange_self_source are mutually exclusive"
        # Control experiment (paper Table 6, single-scale vs multi-scale):
        # restricts photo2sketch_net's key/value at layer i to only that
        # layer's own proxy p~^i instead of the full concatenated proxy set.
        # Orthogonal to the exchange_* flags above -- freely combinable.
        self.mapper_single_scale = getattr(cfg, 'mapper_single_scale', False)
        # Capacity-matched no-exchange control (Run C): reuses photo2sketch_net
        # as a per-layer self-attention refine on cross_prompts_sketch[i] --
        # q=sk (full gradient), k=v=sk.detach() (no LKP, no proxy, no photo
        # tensor, no normalization on k/v, k/v detached). Only meaningful
        # when the exchange itself is off.
        self.sketch_self_refine = getattr(cfg, 'sketch_self_refine', False)
        assert not self.sketch_self_refine or self.disable_exchange, \
            "--sketch_self_refine requires --disable_exchange"
        # Run D: same as Run C but k/v = LayerNorm(sk).detach() -- adds
        # normalization on the k/v side only, still detached, still no
        # compression (no LKP/proxy). Mutually exclusive with Run C (both
        # reuse the same forward slot -- pick exactly one self-refine scheme).
        self.sketch_self_refine_ln = getattr(cfg, 'sketch_self_refine_ln', False)
        assert not self.sketch_self_refine_ln or self.disable_exchange, \
            "--sketch_self_refine_ln requires --disable_exchange"
        assert not (self.sketch_self_refine and self.sketch_self_refine_ln), \
            "--sketch_self_refine and --sketch_self_refine_ln are mutually exclusive"
        # L1: init scheme for photo_proxy_token/sketch_proxy_token (the LKP
        # query). 'randn' = std=1.0 (original code, kept as default for exact
        # backward compat), 'small' = std=0.02 (matches every other tensor in
        # this file), 'mean' = per-layer mean of that layer's own cross_prompts
        # at init time (see _make_proxy_tokens).
        self.proxy_init = getattr(cfg, 'proxy_init', 'randn')
        assert self.proxy_init in ('randn', 'small', 'mean'), \
            f"--proxy_init must be one of randn/small/mean, got {self.proxy_init!r}"
        # L2': number of proxy tokens produced per layer by the LKP (default 1
        # = original behavior). Mapper then sees cross_layer * n_proxy tokens
        # as k/v instead of cross_layer. Orthogonal to proxy_init.
        self.n_proxy = getattr(cfg, 'n_proxy', 1)
        assert self.n_proxy >= 1, "--n_proxy must be >= 1"

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
        # --disable_exchange is a CLEAN ablation: the exchange modules are not
        # built at all, so they never reach the optimizer and never appear in
        # the checkpoint. Building-but-not-calling them (the old behaviour) left
        # ~39M params with .grad permanently None -- they inflated every
        # "trainable params" count while contributing nothing, and Adam still
        # allocated moment buffers for them.
        #
        # The only exception is photo2sketch_net under the self-refine runs:
        # forward() still calls it there (see the sketch_self_refine branches),
        # so it must exist even though disable_exchange is set.
        build_exchange = not self.disable_exchange
        needs_selfrefine_mapper = self.sketch_self_refine or self.sketch_self_refine_ln

        # BUILD-THEN-DISCARD. Every module below is CONSTRUCTED unconditionally,
        # in the original order, but only ASSIGNED to self when this run really
        # uses it. Two properties must hold at once and they pull against each
        # other:
        #
        #   clean ablation -- an unassigned module is not an attribute, so it
        #       never reaches named_parameters(), the optimizer, or the
        #       checkpoint. --disable_exchange really does train 0 exchange
        #       params, not "39M params that happen to get no gradient".
        #
        #   seed parity -- construction consumes the global RNG (Linear /
        #       MultiheadAttention init, _make_proxy_tokens' randn). Skipping it
        #       would shift the stream for everything built afterwards, and
        #       text_prompt_photo / text_prompt_sketch are built AFTER this
        #       learner: their cross_prompts_text[1:] (32,768 params) would get
        #       a different draw. --disable_exchange would then differ from its
        #       baseline by BOTH the ablation and a reroll of 20% of the
        #       trainable params -- a confounded comparison. Constructing and
        #       dropping costs a few ms and keeps the two runs seed-identical.
        #
        # _get_clones uses deepcopy, so only the prototype draws from the RNG.
        if self.cross_layer > 0:
            photo2sketch_net = CrossPromptAttention(hidden_size=s_dim, encoder_hidden_size=p_dim, num_attention_heads=8)

            attn_pooling_photo = AttentionPooling(hidden_size=p_dim, num_attention_heads=8)
            attn_pooling_photo_nets = _get_clones(attn_pooling_photo, self.cross_layer)

            photo_proxy_token = _make_proxy_tokens(
                self.proxy_init, self.n_proxy, p_dim, dtype,
                [self.cross_prompts_photo[i] for i in range(self.cross_layer)],
            )

            free_source = None
            if self.exchange_free_source:
                # Same shape as proxy_photo_prompts (P~_photo) after torch.cat:
                # (cross_layer, p_dim). Independent of ctx_photo/attn_pooling_photo
                # -- Mapper param count (photo2sketch_net) is unaffected.
                free_source_init = torch.empty(self.cross_layer, p_dim, dtype=dtype)
                nn.init.normal_(free_source_init, std=0.02)
                free_source = nn.Parameter(free_source_init)

            # forward() still calls photo2sketch_net in the self-refine branches,
            # so it survives --disable_exchange there.
            if build_exchange or needs_selfrefine_mapper:
                self.photo2sketch_net = photo2sketch_net
            if build_exchange:
                self.attn_pooling_photo_nets = attn_pooling_photo_nets
                self.photo_proxy_token = photo_proxy_token
                if free_source is not None:
                    self.free_source = free_source

            if self.sketch_self_refine_ln:
                # Run D: LayerNorm applied to the k/v side only, standard init
                # (weight=1, bias=0 -- nn.LayerNorm default, no custom init).
                self.ln_selfrefine = nn.LayerNorm(s_dim)

        n_deep = self.prompt_depth - self.cross_layer
        if n_deep > 0:
            sketch2photo_net = CrossPromptAttention(hidden_size=p_dim, encoder_hidden_size=s_dim, num_attention_heads=8)

            attn_pooling_sketch = AttentionPooling(hidden_size=s_dim, num_attention_heads=8)
            attn_pooling_sketch_nets = _get_clones(attn_pooling_sketch, n_deep)

            sketch_proxy_token = _make_proxy_tokens(
                self.proxy_init, self.n_proxy, s_dim, dtype,
                [self.cross_prompts_sketch[i] for i in range(self.cross_layer, self.prompt_depth)],
            )

            if build_exchange:
                self.sketch2photo_net = sketch2photo_net
                self.attn_pooling_sketch_nets = attn_pooling_sketch_nets
                self.sketch_proxy_token = sketch_proxy_token
        ######## Knowledge mapper end ########

        self._freeze_gradientless_params()

    def _freeze_gradientless_params(self):
        """Mark every parameter that provably cannot receive gradient as frozen.

        Companion to the --disable_exchange "clean ablation" above, for the flags
        where the module CANNOT simply be dropped: --exchange_detach_source /
        --exchange_self_source still CALL attn_pooling_photo_nets and feed its
        output into the Mapper (only .detach() cuts the gradient), and
        --sketch_self_refine_ln still applies ln_selfrefine to the k/v side.
        Deleting those modules would change the numerics, i.e. a different
        experiment -- so they stay in the graph, but requires_grad=False keeps
        them out of the optimizer and out of every "trainable params" count.

        Net effect across all flags: declared trainable == actually trained.
        """
        dead = []
        if self.cross_layer > 0 and (self.exchange_detach_source
                                     or self.exchange_self_source
                                     or self.exchange_free_source):
            # Output detached before the Mapper (src/hicropl.py:644-647), or the
            # module is bypassed entirely under --exchange_free_source.
            dead += [self.attn_pooling_photo_nets, self.photo_proxy_token]
        if self.sketch_self_refine_ln:
            # Applied only on the k/v side, which is detached.
            dead.append(self.ln_selfrefine)

        for module in dead:
            for p in module.parameters():
                p.requires_grad_(False)

    def forward(self):
        # Local mutable copies -- entries get REPLACED here, never the stored
        # nn.Parameter itself, so gradients flow correctly into the mapper/LKP
        # networks (matches the original HiCroPL exactly; see class docstring).
        current_photo_prompts = list(self.cross_prompts_photo)
        current_sketch_prompts = list(self.cross_prompts_sketch)

        ######## Photo -> Sketch mapping (shallow layers [0, cross_layer)) ########
        if not self.disable_exchange and self.cross_layer > 0:
            if self.exchange_free_source:
                # Branch B: source k/v is an independent learned parameter,
                # unrelated to photo -- attn_pooling_photo is skipped entirely
                # (its output would be discarded anyway).
                proxy_photo_flat = self.free_source
            else:
                # Control experiment: same LKP module (attn_pooling_photo_nets),
                # same photo_proxy_token query, but the sequence fed in is
                # cross_prompts_sketch[i] instead of cross_prompts_photo[i] --
                # no photo tensor enters this block at all when self_source is on.
                pooling_source = current_sketch_prompts if self.exchange_self_source else current_photo_prompts
                proxy_photo_tokens = []
                for i in range(self.cross_layer):
                    photo_proxy_token = self.attn_pooling_photo_nets[i](
                        token_query=self.photo_proxy_token[i],
                        sequence_key=pooling_source[i],
                        sequence_value=pooling_source[i],
                    )
                    proxy_photo_tokens.append(photo_proxy_token)
                proxy_photo_prompts = torch.cat(proxy_photo_tokens, dim=0)
                proxy_photo_flat = proxy_photo_prompts.view(-1, proxy_photo_prompts.shape[-1])
                if self.exchange_detach_source or self.exchange_self_source:
                    # Branch A (or self_source control): same numeric value,
                    # gradient cut before the Mapper.
                    proxy_photo_flat = proxy_photo_flat.detach()

            if self.mapper_single_scale:
                # Single-scale: same photo2sketch_net module, same query per
                # layer, but key/value restricted to that layer's own proxy
                # p~^i only (shape [1, dim]) -- no cross-layer proxy scope.
                updated_sketch_prompts = []
                for i in range(self.cross_layer):
                    layer_kv = proxy_photo_flat[i:i + 1]  # [1, dim]
                    updated_sketch_prompts.append(
                        self.photo2sketch_net(current_sketch_prompts[i], layer_kv, layer_kv)
                    )
            else:
                sketch_prompts_range = torch.cat(
                    [current_sketch_prompts[i].unsqueeze(0) for i in range(self.cross_layer)], dim=0
                )
                sketch_prompts_flat = sketch_prompts_range.view(-1, sketch_prompts_range.shape[-1])

                updated_sketch_prompts = self.photo2sketch_net(sketch_prompts_flat, proxy_photo_flat, proxy_photo_flat)
                updated_sketch_prompts = updated_sketch_prompts.view(
                    self.cross_layer, -1, updated_sketch_prompts.shape[-1]
                )
            for i in range(self.cross_layer):
                current_sketch_prompts[i] = updated_sketch_prompts[i]
        elif self.sketch_self_refine and self.cross_layer > 0:
            # Run C: photo2sketch_net as a per-layer self-attention refine,
            # q=sk (undetached), k=v=sk.detach() -- no LKP, no proxy, no
            # normalization, nothing from photo. photo2sketch_net still sits
            # in the forward graph and receives real gradient (via q), unlike
            # plain --disable_exchange where it is idle/dead weight.
            for i in range(self.cross_layer):
                sk = current_sketch_prompts[i]
                current_sketch_prompts[i] = self.photo2sketch_net(sk, sk.detach(), sk.detach())
        elif self.sketch_self_refine_ln and self.cross_layer > 0:
            # Run D: same as Run C, but k/v = LayerNorm(sk).detach() --
            # normalization added on the k/v side only, still detached, still
            # no compression (no LKP/proxy).
            for i in range(self.cross_layer):
                sk = current_sketch_prompts[i]
                sk_kv = self.ln_selfrefine(sk).detach()
                current_sketch_prompts[i] = self.photo2sketch_net(sk, sk_kv, sk_kv)
        ######## Photo -> Sketch end ########

        ######## Sketch -> Photo mapping (deep layers [cross_layer, prompt_depth)) ########
        n_deep = self.prompt_depth - self.cross_layer
        if not self.disable_exchange and n_deep > 0:
            proxy_sketch_tokens = []
            for i in range(self.cross_layer, self.prompt_depth):
                sketch_proxy_token = self.attn_pooling_sketch_nets[i - self.cross_layer](
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


def _build_description_prompts(raw_names, clean_names, descriptions, desc_pos, n_ctx):
    """Assemble the description prompt string for every class, budget-checked.

    The leading `n_ctx` "X" tokens are LOAD-BEARING placeholders, not decoration:
    SimpleTextPromptLearner drops embedding[:, 1:1+n_ctx] and puts the learnable
    ctx there, so whatever sits in that window is deleted. Without the padding,
    ctx would erase the first words of the description instead.

    Budget: 1(SOS) + n_ctx + L_d + L_cls + 1(EOT) <= 77. Checked with the raw
    tokenizer (no length cap) BEFORE clip.tokenize is called, because
    clip.tokenize(truncate=False) raises a bare RuntimeError naming no class.
    """
    from src.clip.clip import _tokenizer

    if desc_pos not in ('V1', 'V2'):
        raise ValueError(f"desc_pos must be 'V1' or 'V2', got {desc_pos!r}")
    placeholder = " ".join(["X"] * n_ctx)

    prompts, offenders = [], []
    for raw, clean in zip(raw_names, clean_names):
        d = descriptions[raw].strip().rstrip(".").lower()
        if desc_pos == 'V1':
            prompt = f"{placeholder} {d}, a {clean}."
        else:
            prompt = f"{placeholder} a {clean}, {d}."
        n_total = 2 + len(_tokenizer.encode(prompt))   # + SOS + EOT
        if n_total > 77:
            offenders.append((raw, len(_tokenizer.encode(d)),
                              len(_tokenizer.encode(f"a {clean}.")), n_total))
        prompts.append(prompt)

    if offenders:
        rows = "\n".join(
            f"    {name}: L_d={ld} L_cls={lc} -> 1 + {n_ctx} + L_d + L_cls + 1 = {tot} > 77"
            for name, ld, lc, tot in offenders
        )
        raise ValueError(
            f"{len(offenders)} class(es) do not fit CLIP's 77-token context at "
            f"n_ctx={n_ctx}, desc_pos={desc_pos}:\n{rows}\n"
            f"Shorten those descriptions or lower n_ctx."
        )
    return prompts


class SimpleTextPromptLearner(nn.Module):
    """Minimal text-only prompt learner: prepares tokenized prompts and text prompt tensors.

    Matches the outputs needed by TextEncoder but does not perform cross-modal mapping.
    """

    def __init__(self, cfg, classnames, clip_model, descriptions=None, desc_pos="V1"):
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

        # P2: the branch above drops ctx_init SILENTLY when n_ctx > 4 (see the
        # `if ctx_init and (n_ctx) <= 4` condition earlier in this __init__).
        # Warn only -- behaviour deliberately left as it is.
        if ctx_init and n_ctx > 4:
            print(f"[WARN] SimpleTextPromptLearner: ctx_init={ctx_init!r} is IGNORED because "
                  f"n_ctx={n_ctx} > 4. ctx falls back to normal(std=0.02) init and the prompt "
                  f"prefix becomes {' '.join(['X'] * n_ctx)!r}. Behaviour unchanged.")

        # register token prefix/suffix buffers for class prompts
        #
        # All THREE buffers below are derived from the same `prompts` list. They
        # must stay in sync: token_prefix/token_suffix feed the embedding that
        # goes into the encoder, while tokenized_prompts is what TextEncoder
        # runs argmax over to locate EOT for pooling. Regenerating one without
        # the others produces a valid-shaped tensor pooled at the wrong
        # position -- no exception, just a wrong feature (see the P1 assert).
        raw_names = list(classnames)   # keys exactly as they appear in the JSON
        classnames = [name.replace("_", " ") for name in classnames]
        if descriptions is None:
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            prompts = _build_description_prompts(
                raw_names, classnames, descriptions, desc_pos, n_ctx)
        from src.clip import clip as _clip
        tokenized_prompts = torch.cat([_clip.tokenize(p) for p in prompts]).to(clip_model.token_embedding.weight.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        # P1: EOT must be the last real token, because TextEncoder pools at
        # tokenized_prompts.argmax(-1) and that index is only the EOT position
        # while this holds. A stale tokenized_prompts fails here instead of
        # silently pooling mid-sentence.
        eot_idx = tokenized_prompts.argmax(dim=-1)
        n_real = (tokenized_prompts != 0).sum(dim=-1) - 1
        assert torch.equal(eot_idx, n_real), (
            "P1: tokenized_prompts.argmax(-1) is not the last non-pad position -- the buffer "
            f"does not describe the strings it was built from. argmax={eot_idx[:5].tolist()} "
            f"vs last-real={n_real[:5].tolist()}"
        )

        # P8: the ctx window must contain only placeholders. If real content
        # sits at positions 1..n_ctx it is DELETED (that slice never reaches the
        # encoder), which no shape check would catch.
        if descriptions is not None:
            from src.clip.clip import _tokenizer
            x_id = _tokenizer.encode("X")[0]
            window = tokenized_prompts[:, 1:1 + n_ctx]
            bad = (window != x_id).any(dim=-1).nonzero().flatten()
            assert bad.numel() == 0, (
                f"P8: {bad.numel()} prompt(s) do not start with {n_ctx} 'X' placeholders, so ctx "
                f"would overwrite real content. First offender: {raw_names[int(bad[0])]!r} -> "
                f"window ids {window[int(bad[0])].tolist()} (expected all {x_id})"
            )

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

    