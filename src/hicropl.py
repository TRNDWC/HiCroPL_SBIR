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


class QuickGELU(nn.Module):
    """Fast GELU approximation (same as in CLIP)."""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def _get_clones(module, N):
    """Create N deep copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# =============================================================================
# 1. Layer-specific Knowledge Proxy (LKP)
# =============================================================================

class AttentionPooling(nn.Module):
    """Layer-specific Knowledge Proxy (LKP).
    
    Compresses multiple prompt tokens into a single proxy token
    via cross-attention pooling. Used to create compact representations
    of per-layer prompts before cross-modal transfer.
    """

    def __init__(self, hidden_size, num_attention_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_attention_heads
        )
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, token_query, sequence_key, sequence_value):
        """
        Args:
            token_query:    [1, hidden_size] - learnable proxy token
            sequence_key:   [n_ctx, hidden_size] - prompt tokens (Key)
            sequence_value: [n_ctx, hidden_size] - prompt tokens (Value)
        Returns:
            [1, hidden_size] - compressed proxy token
        """
        token_query = token_query + self.attn(
            self.ln_1(token_query),
            self.ln_1(sequence_key),
            self.ln_1(sequence_value),
            need_weights=False,
        )[0]
        token_query = self.ln_2(token_query)
        return token_query


# =============================================================================
# 2. Multi-scale Knowledge Mapper
# =============================================================================

class CrossPromptAttention(nn.Module):
    """Multi-scale Knowledge Mapper.
    
    Projects cross-modal knowledge from source modality proxy tokens
    into target modality prompts using cross-attention + FFN.
    Handles dimension mismatch between text (512) and visual (768) spaces.
    """

    def __init__(self, hidden_size, encoder_hidden_size, num_attention_heads=8):
        """
        Args:
            hidden_size: dimension of target modality (Q)
            encoder_hidden_size: dimension of source modality (K, V)
            num_attention_heads: number of attention heads
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_attention_heads
        )
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
        """
        Args:
            q: [n_target, hidden_size] - target modality prompts
            k: [n_source, encoder_hidden_size] - source modality proxy tokens
            v: [n_source, encoder_hidden_size] - source modality proxy tokens
        Returns:
            [n_target, hidden_size] - updated target prompts
        """
        q_proj = self.linear_q(q)
        k_proj = self.linear_k(k)
        v_proj = self.linear_v(v)
        q_proj = q_proj + self.attn(
            self.ln_1(q_proj),
            self.ln_1(k_proj),
            self.ln_1(v_proj),
            need_weights=False,
        )[0]
        q_proj = q_proj + self.ffn(self.ln_2(q_proj))
        return q_proj


# =============================================================================
# 3. Text Encoder with Deep Prompt Injection
# =============================================================================

class TextEncoder(nn.Module):
    """Wraps CLIP text transformer to support per-layer deep prompt injection.

    Uses the upstream HiCroPL list contract: passes [x, deep_prompts_text] to
    the nn.Sequential of ResidualAttentionBlock_HiCroPL blocks, which replace
    context tokens internally at each layer (using self.i-1 indexing).
    """

    def __init__(self, clip_model):
        super().__init__()
        self.transformer_resblocks = clip_model.transformer.resblocks
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, deep_prompts_text):
        """
        Args:
            prompts: [n_cls, 77, ctx_dim] - first layer text input 
                     (SOS + learnable_ctx + class_name + EOS + padding)
            tokenized_prompts: [n_cls, 77] - tokenized prompts (for EOT position)
            deep_prompts_text: list of L-1 tensors [n_ctx, ctx_dim] for layers 1..L-1
        Returns:
            [n_cls, embed_dim] - text features (e.g. [n_cls, 512])
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND [77, n_cls, ctx_dim]

        # Use upstream list contract: pass [x, deep_prompts_text] through nn.Sequential.
        # Each ResidualAttentionBlock_HiCroPL receives this list and replaces context
        # tokens internally at the correct layer (using self.i-1 index), then returns
        # [x_updated, deep_prompts_text] for the next block.
        result = self.transformer_resblocks([x, deep_prompts_text])
        x = result[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )
        return x


# =============================================================================
# 4. Visual Encoder with Deep Prompt Injection
# =============================================================================

class VisualEncoder(nn.Module):
    """Wraps CLIP VisionTransformer_HiCroPL for deep prompt injection.
    
    Delegates directly to VisionTransformer_HiCroPL.forward(x, img_prompts,
    cross_prompts_visual_deeper), which passes [x, cross_prompts] as a list to
    the transformer so each ResidualAttentionBlock_HiCroPL can replace visual
    prompt tokens internally — the correct upstream CLIP-level contract.
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


# =============================================================================
# 5. Cross-Modal Prompt Learner (Bidirectional Flow)
# =============================================================================

class CrossModalPromptLearner(nn.Module):
    """Manages one text-visual bidirectional knowledge flow.
    
    For ZS-SBIR, create 2 instances:
        - CrossModalPromptLearner(...) for text <-> photo
        - CrossModalPromptLearner(...) for text <-> sketch
    
    Bidirectional flow:
        Early layers (0..cross_layer-1): Text -> Visual
            Text prompts are compressed via LKP, then Knowledge Mapper
            injects text semantics into visual prompts.
        Later layers (cross_layer..prompt_depth-1): Visual -> Text
            Visual prompts are compressed via LKP, then Knowledge Mapper
            injects visual object info back into text prompts.
    
    Note: Following the original HiCroPL, prompts are updated via .data.copy_()
    which detaches the gradient. The mapper/LKP act as fixed transformations;
    learning happens in the prompt parameters themselves via encoder gradients.
    WARNING: In-place data copy might introduce subtle non-determinism across
    different hardware or CUDA versions due to floating point accumulation.
    """

    def __init__(
        self,
        clip_model,
        n_ctx=4,
        prompt_depth=9,
        cross_layer=4,
        ctx_init="a photo of a",
        use_fp16=True,
    ):
        """
        Args:
            clip_model: loaded CLIP model (for token_embedding and dtype)
            n_ctx: number of learnable context tokens per layer
            prompt_depth: total number of layers with prompts (L)
            cross_layer: layer index where flow switches from T->V to V->T (k)
            ctx_init: initial text for first layer prompt (max n_ctx words)
            use_fp16: whether to use fp16 for mapper/LKP networks
        """
        super().__init__()
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512 for ViT-B/32
        v_dim = clip_model.visual.conv1.weight.shape[0]  # 768 for ViT-B/32

        self.prompt_depth = prompt_depth
        self.cross_layer = cross_layer
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.v_dim = v_dim
        self.ctx_init_text = ctx_init

        # Reference to token_embedding (frozen, not trained)
        self.token_embedding = clip_model.token_embedding

        # ---- Text Prompts: L layers x [n_ctx, ctx_dim] ----
        if ctx_init and n_ctx <= 4:
            from src.clip import clip as _clip
            prompt = _clip.tokenize(ctx_init).to(clip_model.token_embedding.weight.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.cross_prompts_text = nn.ParameterList(
            [nn.Parameter(ctx_vectors)]
            + [
                nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=self.dtype))
                for _ in range(prompt_depth - 1)
            ]
        )
        for p in self.cross_prompts_text[1:]:
            nn.init.normal_(p, std=0.02)

        # ---- Visual Prompts: L layers x [n_ctx, v_dim] ----
        # Ablation path: initialize from one shared base tensor across depths.
        visual_vectors = torch.empty(n_ctx, v_dim, dtype=self.dtype)
        nn.init.normal_(visual_vectors, std=0.02)
        self.cross_prompts_visual = nn.ParameterList(
            [nn.Parameter(visual_vectors) for _ in range(prompt_depth)]
        )

        # ---- Knowledge Mapper Networks ----
        self.text2visual_net = CrossPromptAttention(
            hidden_size=v_dim, encoder_hidden_size=ctx_dim, num_attention_heads=8
        )
        self.visual2text_net = CrossPromptAttention(
            hidden_size=ctx_dim, encoder_hidden_size=v_dim, num_attention_heads=8
        )

        # ---- LKP Networks ----
        # T->V: compress text prompts at early layers
        attn_text = AttentionPooling(hidden_size=ctx_dim, num_attention_heads=8)
        self.attn_pooling_text_nets = _get_clones(attn_text, cross_layer)

        # V->T: compress visual prompts at later layers
        attn_visual = AttentionPooling(hidden_size=v_dim, num_attention_heads=8)
        self.attn_pooling_visual_nets = _get_clones(
            attn_visual, prompt_depth - cross_layer
        )

        # ---- Proxy Tokens ----
        self.text_proxy_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, ctx_dim, dtype=self.dtype))
                for _ in range(cross_layer)
            ]
        )
        self.visual_proxy_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, v_dim, dtype=self.dtype))
                for _ in range(prompt_depth - cross_layer)
            ]
        )

        # Convert to fp16 if needed
        if use_fp16:
            self.text2visual_net = self.text2visual_net.half()
            self.visual2text_net = self.visual2text_net.half()
            self.attn_pooling_text_nets = self.attn_pooling_text_nets.half()
            self.attn_pooling_visual_nets = self.attn_pooling_visual_nets.half()

    def construct_prompts(self, ctx, prefix, suffix):
        """Build text input: [SOS, ctx_tokens, class_name_tokens..., EOS, padding]."""
        return torch.cat([prefix, ctx, suffix], dim=1)

    def _text_to_visual_flow(self):
        """Early layers (0..cross_layer-1): Text semantics -> Visual prompts."""
        # LKP: compress each early-layer text prompt into a proxy token
        proxy_text_tokens = []
        for i in range(self.cross_layer):
            proxy = self.attn_pooling_text_nets[i](
                token_query=self.text_proxy_tokens[i],
                sequence_key=self.cross_prompts_text[i],
                sequence_value=self.cross_prompts_text[i],
            )
            proxy_text_tokens.append(proxy)
        proxy_text = torch.cat(proxy_text_tokens, dim=0)  # [cross_layer, ctx_dim]

        # Flatten visual prompts for early layers
        visual_flat = torch.cat(
            [self.cross_prompts_visual[i].unsqueeze(0) for i in range(self.cross_layer)],
            dim=0,
        ).view(-1, self.v_dim)  # [cross_layer * n_ctx, v_dim]
        proxy_text_flat = proxy_text.view(-1, self.ctx_dim)  # [cross_layer, ctx_dim]

        # Knowledge Mapper: inject text info into visual prompts
        updated_visual = self.text2visual_net(visual_flat, proxy_text_flat, proxy_text_flat)
        updated_visual = updated_visual.view(self.cross_layer, self.n_ctx, self.v_dim)

        # Update visual prompts (detached, following original HiCroPL)
        for i in range(self.cross_layer):
            self.cross_prompts_visual[i].data.copy_(updated_visual[i])

    def _visual_to_text_flow(self):
        """Later layers (cross_layer..prompt_depth-1): Visual info -> Text prompts."""
        n_later = self.prompt_depth - self.cross_layer

        # LKP: compress each later-layer visual prompt into a proxy token
        proxy_visual_tokens = []
        for i in range(self.cross_layer, self.prompt_depth):
            idx = i - self.cross_layer
            proxy = self.attn_pooling_visual_nets[idx](
                token_query=self.visual_proxy_tokens[idx],
                sequence_key=self.cross_prompts_visual[i],
                sequence_value=self.cross_prompts_visual[i],
            )
            proxy_visual_tokens.append(proxy)
        proxy_visual = torch.cat(proxy_visual_tokens, dim=0)  # [n_later, v_dim]

        # Flatten text prompts for later layers
        text_flat = torch.cat(
            [
                self.cross_prompts_text[i].unsqueeze(0)
                for i in range(self.cross_layer, self.prompt_depth)
            ],
            dim=0,
        ).view(-1, self.ctx_dim)  # [n_later * n_ctx, ctx_dim]
        proxy_visual_flat = proxy_visual.view(-1, self.v_dim)  # [n_later, v_dim]

        # Knowledge Mapper: inject visual info into text prompts
        updated_text = self.visual2text_net(text_flat, proxy_visual_flat, proxy_visual_flat)
        updated_text = updated_text.view(n_later, self.n_ctx, self.ctx_dim)

        # Update text prompts (detached, following original HiCroPL)
        for i in range(self.cross_layer, self.prompt_depth):
            self.cross_prompts_text[i].data.copy_(updated_text[i - self.cross_layer])

    def forward(self, classnames):
        """Perform bidirectional knowledge flow and return all prompts.
        
        Args:
            classnames: list of class name strings (e.g. ["cat", "dog", ...])
        Returns:
            text_input: [n_cls, 77, ctx_dim] - full first-layer text sequence
            tokenized_prompts: [n_cls, 77] - for EOT position lookup
            first_visual_prompt: [n_ctx, v_dim] - first-layer visual prompt
            deeper_text_prompts: list of L-1 tensors [n_ctx, ctx_dim]
            deeper_visual_prompts: list of L-1 tensors [n_ctx, v_dim]
        """
        n_cls = len(classnames)
        device = self.cross_prompts_text[0].device

        # ---- Construct text input for first layer ----
        from src.clip import clip as _clip
        classnames_clean = [name.replace("_", " ") for name in classnames]
        raw_prompts = [
            self.ctx_init_text + " " + name + "." for name in classnames_clean
        ]
        tokenized_prompts = torch.cat([_clip.tokenize(p) for p in raw_prompts])
        tokenized_prompts = tokenized_prompts.to(device)

        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)

        prefix = embedding[:, :1, :]               # SOS
        suffix = embedding[:, 1 + self.n_ctx :, :]  # class + EOS + padding

        ctx = self.cross_prompts_text[0]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

        text_input = self.construct_prompts(ctx, prefix, suffix)

        # ---- Bidirectional Knowledge Flow ----
        self._text_to_visual_flow()   # T -> V (early layers)
        self._visual_to_text_flow()   # V -> T (later layers)

        # ---- Collect deeper prompts (layers 1..L-1) ----
        deeper_text_prompts = [
            self.cross_prompts_text[i] for i in range(1, self.prompt_depth)
        ]
        deeper_visual_prompts = [
            self.cross_prompts_visual[i] for i in range(1, self.prompt_depth)
        ]

        return (
            text_input,
            tokenized_prompts,
            self.cross_prompts_visual[0],
            deeper_text_prompts,
            deeper_visual_prompts,
        )
