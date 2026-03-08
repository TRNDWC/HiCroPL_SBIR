"""
SharedTextDualVisualPromptLearner - 1 Text Shared + 2 Visual Branches
This module extends HiCroPL for ZS-SBIR with shared text prompts.

Architecture:
    - 1 Text Prompt set (shared for both photo and sketch)
    - 2 Visual Prompt sets (separate for photo and sketch)  
    - Bidirectional flows: T ↔ V_photo and T ↔ V_sketch
"""

import torch
import torch.nn as nn
from src.hicropl import (
    AttentionPooling,
    CrossPromptAttention,
    _get_clones
)


class SharedTextDualVisualPromptLearner(nn.Module):
    """Shared text prompts with dual visual branches for photo and sketch.
    
    This replaces the original dual CrossModalPromptLearner design where
    each modality had separate text and visual prompts. Now:
    - Text prompts are shared between photo and sketch
    - Visual prompts remain separate
    - Bidirectional knowledge flows: T ↔ V_photo and T ↔ V_sketch
    
    Benefits:
    - Encourages unified semantic understanding across modalities
    - Reduces redundancy in text prompt learning
    - Maintains flexibility in visual feature adaptation
    """

    def __init__(
        self,
        clip_model,
        n_ctx=4,
        prompt_depth=9,
        cross_layer=4,
        ctx_init="a photo or a sketch of a",
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

        # ======================================================================
        # 1. SHARED TEXT PROMPTS (L layers)
        # ======================================================================
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

        # ======================================================================
        # 2. DUAL VISUAL PROMPTS (L layers each)
        # ======================================================================
        # Photo visual prompts
        self.cross_prompts_visual_photo = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_ctx, v_dim, dtype=self.dtype))
                for _ in range(prompt_depth)
            ]
        )
        for p in self.cross_prompts_visual_photo:
            nn.init.normal_(p, std=0.02)

        # Sketch visual prompts
        self.cross_prompts_visual_sketch = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_ctx, v_dim, dtype=self.dtype))
                for _ in range(prompt_depth)
            ]
        )
        for p in self.cross_prompts_visual_sketch:
            nn.init.normal_(p, std=0.02)

        # ======================================================================
        # 3. KNOWLEDGE MAPPERS
        # ======================================================================
        # Shared T->V mapper (used for both photo and sketch)
        self.text2visual_net = CrossPromptAttention(
            hidden_size=v_dim, encoder_hidden_size=ctx_dim, num_attention_heads=8
        )
        
        # V->T mappers (separate for photo and sketch)
        self.visual2text_net_photo = CrossPromptAttention(
            hidden_size=ctx_dim, encoder_hidden_size=v_dim, num_attention_heads=8
        )
        self.visual2text_net_sketch = CrossPromptAttention(
            hidden_size=ctx_dim, encoder_hidden_size=v_dim, num_attention_heads=8
        )

        # ======================================================================
        # 4. TEXT LKP NETWORKS (for T->V flow at early layers)
        # ======================================================================
        attn_text = AttentionPooling(hidden_size=ctx_dim, num_attention_heads=8)
        self.attn_pooling_text_nets = _get_clones(attn_text, cross_layer)
        
        # Text proxy tokens for T->V compression
        self.text_proxy_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, ctx_dim, dtype=self.dtype))
                for _ in range(cross_layer)
            ]
        )

        # ======================================================================
        # 5. VISUAL LKP NETWORKS (for V->T flow at later layers)
        # ======================================================================
        n_later = prompt_depth - cross_layer
        
        # Photo visual LKP
        attn_visual_photo = AttentionPooling(hidden_size=v_dim, num_attention_heads=8)
        self.attn_pooling_visual_photo_nets = _get_clones(attn_visual_photo, n_later)
        
        self.visual_photo_proxy_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, v_dim, dtype=self.dtype))
                for _ in range(n_later)
            ]
        )
        
        # Sketch visual LKP
        attn_visual_sketch = AttentionPooling(hidden_size=v_dim, num_attention_heads=8)
        self.attn_pooling_visual_sketch_nets = _get_clones(attn_visual_sketch, n_later)
        
        self.visual_sketch_proxy_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, v_dim, dtype=self.dtype))
                for _ in range(n_later)
            ]
        )

        # ======================================================================
        # 6. Convert to FP16 if needed
        # ======================================================================
        if use_fp16:
            self.text2visual_net = self.text2visual_net.half()
            self.visual2text_net_photo = self.visual2text_net_photo.half()
            self.visual2text_net_sketch = self.visual2text_net_sketch.half()
            self.attn_pooling_text_nets = self.attn_pooling_text_nets.half()
            self.attn_pooling_visual_photo_nets = self.attn_pooling_visual_photo_nets.half()
            self.attn_pooling_visual_sketch_nets = self.attn_pooling_visual_sketch_nets.half()

    def construct_prompts(self, ctx, prefix, suffix):
        """Build text input: [SOS, ctx_tokens, class_name_tokens..., EOS, padding]."""
        return torch.cat([prefix, ctx, suffix], dim=1)

    def _text_to_visual_flow(self):
        """Early layers (0..cross_layer-1): Text semantics -> Visual prompts.
        
        Shared text prompts are compressed and injected into BOTH photo and sketch
        visual prompts independently.
        """
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
        proxy_text_flat = proxy_text.view(-1, self.ctx_dim)  # [cross_layer, ctx_dim]

        # ---- Update Photo Visual Prompts ----
        visual_photo_flat = torch.cat(
            [self.cross_prompts_visual_photo[i].unsqueeze(0) for i in range(self.cross_layer)],
            dim=0,
        ).view(-1, self.v_dim)  # [cross_layer * n_ctx, v_dim]

        updated_visual_photo = self.text2visual_net(visual_photo_flat, proxy_text_flat, proxy_text_flat)
        updated_visual_photo = updated_visual_photo.view(self.cross_layer, self.n_ctx, self.v_dim)

        for i in range(self.cross_layer):
            self.cross_prompts_visual_photo[i].data.copy_(updated_visual_photo[i])

        # ---- Update Sketch Visual Prompts ----
        visual_sketch_flat = torch.cat(
            [self.cross_prompts_visual_sketch[i].unsqueeze(0) for i in range(self.cross_layer)],
            dim=0,
        ).view(-1, self.v_dim)  # [cross_layer * n_ctx, v_dim]

        updated_visual_sketch = self.text2visual_net(visual_sketch_flat, proxy_text_flat, proxy_text_flat)
        updated_visual_sketch = updated_visual_sketch.view(self.cross_layer, self.n_ctx, self.v_dim)

        for i in range(self.cross_layer):
            self.cross_prompts_visual_sketch[i].data.copy_(updated_visual_sketch[i])

    def _visual_to_text_flow(self):
        """Later layers (cross_layer..prompt_depth-1): Visual info -> Text prompts.
        
        Both photo and sketch visual prompts contribute to updating shared text prompts.
        Text prompts are updated with merged knowledge from both modalities.
        """
        n_later = self.prompt_depth - self.cross_layer

        # ---- LKP: Compress Photo Visual prompts ----
        proxy_visual_photo_tokens = []
        for i in range(self.cross_layer, self.prompt_depth):
            idx = i - self.cross_layer
            proxy = self.attn_pooling_visual_photo_nets[idx](
                token_query=self.visual_photo_proxy_tokens[idx],
                sequence_key=self.cross_prompts_visual_photo[i],
                sequence_value=self.cross_prompts_visual_photo[i],
            )
            proxy_visual_photo_tokens.append(proxy)
        proxy_visual_photo = torch.cat(proxy_visual_photo_tokens, dim=0)  # [n_later, v_dim]

        # ---- LKP: Compress Sketch Visual prompts ----
        proxy_visual_sketch_tokens = []
        for i in range(self.cross_layer, self.prompt_depth):
            idx = i - self.cross_layer
            proxy = self.attn_pooling_visual_sketch_nets[idx](
                token_query=self.visual_sketch_proxy_tokens[idx],
                sequence_key=self.cross_prompts_visual_sketch[i],
                sequence_value=self.cross_prompts_visual_sketch[i],
            )
            proxy_visual_sketch_tokens.append(proxy)
        proxy_visual_sketch = torch.cat(proxy_visual_sketch_tokens, dim=0)  # [n_later, v_dim]

        # ---- Flatten text prompts for later layers ----
        text_flat = torch.cat(
            [
                self.cross_prompts_text[i].unsqueeze(0)
                for i in range(self.cross_layer, self.prompt_depth)
            ],
            dim=0,
        ).view(-1, self.ctx_dim)  # [n_later * n_ctx, ctx_dim]

        # ---- Update text prompts from Photo ----
        proxy_visual_photo_flat = proxy_visual_photo.view(-1, self.v_dim)
        updated_text_from_photo = self.visual2text_net_photo(
            text_flat, proxy_visual_photo_flat, proxy_visual_photo_flat
        )
        updated_text_from_photo = updated_text_from_photo.view(n_later, self.n_ctx, self.ctx_dim)

        # ---- Update text prompts from Sketch ----
        proxy_visual_sketch_flat = proxy_visual_sketch.view(-1, self.v_dim)
        updated_text_from_sketch = self.visual2text_net_sketch(
            text_flat, proxy_visual_sketch_flat, proxy_visual_sketch_flat
        )
        updated_text_from_sketch = updated_text_from_sketch.view(n_later, self.n_ctx, self.ctx_dim)

        # ---- Merge updates from both photo and sketch (average) for shared params ----
        updated_text_merged = (updated_text_from_photo + updated_text_from_sketch) / 2.0

        # ---- Update shared text prompts with merged version ----
        for i in range(self.cross_layer, self.prompt_depth):
            self.cross_prompts_text[i].data.copy_(updated_text_merged[i - self.cross_layer])

    def forward(self, classnames):
        """Perform bidirectional knowledge flow and return all prompts.
        
        Shared text prompts are updated with merged knowledge from both photo and sketch.
        Returns single version of text prompts (shared) and dual visual prompts (separate).
        
        Args:
            classnames: list of class name strings (e.g. ["cat", "dog", ...])
        Returns:
            text_input: [n_cls, 77, ctx_dim] - full first-layer text sequence (shared)
            tokenized_prompts: [n_cls, 77] - for EOT position lookup
            first_visual_prompt_photo: [n_ctx, v_dim] - first-layer photo visual prompt
            deeper_text_prompts: list of L-1 tensors [n_ctx, ctx_dim] (shared)
            deeper_visual_prompts_photo: list of L-1 tensors [n_ctx, v_dim]
            first_visual_prompt_sketch: [n_ctx, v_dim] - first-layer sketch visual prompt
            deeper_visual_prompts_sketch: list of L-1 tensors [n_ctx, v_dim]
        """
        n_cls = len(classnames)
        device = self.cross_prompts_text[0].device

        # ---- Construct text input for first layer (shared) ----
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
        self._text_to_visual_flow()   # T -> V_photo and T -> V_sketch (early layers)
        self._visual_to_text_flow()   # V_photo -> T and V_sketch -> T (later layers)

        # ---- Collect deeper prompts ----
        deeper_text_prompts = [
            self.cross_prompts_text[i] for i in range(1, self.prompt_depth)
        ]
        deeper_visual_prompts_photo = [
            self.cross_prompts_visual_photo[i] for i in range(1, self.prompt_depth)
        ]
        deeper_visual_prompts_sketch = [
            self.cross_prompts_visual_sketch[i] for i in range(1, self.prompt_depth)
        ]

        return (
            text_input,
            tokenized_prompts,
            self.cross_prompts_visual_photo[0],
            deeper_text_prompts,
            deeper_visual_prompts_photo,
            self.cross_prompts_visual_sketch[0],
            deeper_visual_prompts_sketch,
        )
