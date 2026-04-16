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
from types import SimpleNamespace

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
    def __init__(self, cfg=None, clip_model=None, clip_model_distill=None, **kwargs):
        super().__init__()

        if clip_model is None:
            raise ValueError("clip_model is required")

        if cfg is None:
            cfg = SimpleNamespace()

        for key, value in kwargs.items():
            if not hasattr(cfg, key):
                setattr(cfg, key, value)

        def _cfg_get(name, modality, default):
            modality_key = f"{name}_{modality}"
            if hasattr(cfg, modality_key):
                return getattr(cfg, modality_key)
            if hasattr(cfg, name):
                return getattr(cfg, name)
            return default

        self.prompt_depth_photo = _cfg_get('prompt_depth', 'photo', 9)
        self.prompt_depth_sketch = _cfg_get('prompt_depth', 'sketch', 9)
        self.cross_layer_photo = _cfg_get('cross_layer', 'photo', 4)
        self.cross_layer_sketch = _cfg_get('cross_layer', 'sketch', 4)
        self.n_ctx_photo = _cfg_get('n_ctx', 'photo', 4)
        self.n_ctx_sketch = _cfg_get('n_ctx', 'sketch', 4)
        self.ctx_init_text_photo = _cfg_get('ctx_init', 'photo', "a photo of a")
        self.ctx_init_text_sketch = _cfg_get('ctx_init', 'sketch', "a photo of a")
        self.dataset_name_photo = _cfg_get('dataset', 'photo', 'sketchy')
        self.dataset_name_sketch = _cfg_get('dataset', 'sketch', 'sketchy')
        self.prec_photo = _cfg_get('prec', 'photo', "fp32")
        self.prec_sketch = _cfg_get('prec', 'sketch', "fp32")

        assert self.prompt_depth_photo >= 1, "Photo prompt depth should be >=1"
        assert self.prompt_depth_sketch >= 1, "Sketch prompt depth should be >=1"
        assert 0 < self.cross_layer_photo < self.prompt_depth_photo, "Photo cross_layer must be in (0, prompt_depth)"
        assert 0 < self.cross_layer_sketch < self.prompt_depth_sketch, "Sketch cross_layer must be in (0, prompt_depth)"
        
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim if hasattr(clip_model.visual, 'output_dim') else clip_model.visual.conv1.weight.shape[0]
        v_dim = 768
        self.ctx_dim = ctx_dim
        self.v_dim = v_dim

        # Backward-compatible defaults map to photo branch values.
        self.prompt_depth = self.prompt_depth_photo
        self.cross_layer = self.cross_layer_photo
        self.n_ctx = self.n_ctx_photo
        self.ctx_init_text = self.ctx_init_text_photo
        self.dataset_name = self.dataset_name_photo
        self.dtype = dtype
        self.token_embedding = clip_model.token_embedding
        self.clip_model = clip_model
        
        # [SỬA ĐỔI LOAD DATA]: Trong framework SBIR, không có teacher gốc mà dùng chung distill branch
        self.clip_model_distill = clip_model_distill if clip_model_distill is not None else clip_model

        def _init_ctx_vectors(ctx_init_value, n_ctx_value):
            if ctx_init_value and n_ctx_value <= 4:
                ctx_init_clean = ctx_init_value.replace("_", " ")
                from src.clip import clip as _clip
                prompt = _clip.tokenize(ctx_init_clean)
                text_device = clip_model.token_embedding.weight.device
                prompt = prompt.to(text_device)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                return embedding[0, 1: 1 + n_ctx_value, :]

            ctx_vectors = torch.empty(n_ctx_value, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            return ctx_vectors

        def _init_visual_vectors(n_ctx_value):
            visual_vectors = torch.empty(n_ctx_value, v_dim, dtype=dtype)
            nn.init.normal_(visual_vectors, std=0.02)
            return visual_vectors

        ctx_vectors_photo = _init_ctx_vectors(self.ctx_init_text_photo, self.n_ctx_photo)
        ctx_vectors_sketch = _init_ctx_vectors(self.ctx_init_text_sketch, self.n_ctx_sketch)
        visual_vectors_photo = _init_visual_vectors(self.n_ctx_photo)
        visual_vectors_sketch = _init_visual_vectors(self.n_ctx_sketch)

        ######## PHOTO branch params ########
        self.ctx_photo = nn.Parameter(ctx_vectors_photo.clone())
        self.cross_prompts_text_photo = nn.ParameterList(
            [self.ctx_photo] +
            [nn.Parameter(torch.empty(self.n_ctx_photo, 512, dtype=dtype)) for _ in range(self.prompt_depth_photo - 1)]
        )
        for single_para in self.cross_prompts_text_photo[1:]:
            nn.init.normal_(single_para, std=0.02)

        self.cross_prompts_visual_photo = nn.ParameterList(
            [nn.Parameter(visual_vectors_photo.clone()) for _ in range(self.prompt_depth_photo)]
        )

        self.text2visual_net_photo = CrossPromptAttention(hidden_size=v_dim, encoder_hidden_size=ctx_dim, num_attention_heads=8)
        self.visual2text_net_photo = CrossPromptAttention(hidden_size=ctx_dim, encoder_hidden_size=v_dim, num_attention_heads=8)
        self.sketch2visual_net_photo = CrossPromptAttention(hidden_size=v_dim, encoder_hidden_size=v_dim, num_attention_heads=8)
        self.gate_alpha = nn.Parameter(torch.tensor(0.01, dtype=dtype))

        attn_pooling_text_photo = AttentionPooling(hidden_size=ctx_dim, num_attention_heads=8)
        self.attn_pooling_text_nets_photo = _get_clones(attn_pooling_text_photo, self.cross_layer_photo)
        attn_pooling_visual_photo = AttentionPooling(hidden_size=v_dim, num_attention_heads=8)
        self.attn_pooling_visual_nets_photo = _get_clones(attn_pooling_visual_photo, self.prompt_depth_photo - self.cross_layer_photo)

        text_proxy_token_photo = torch.randn(1, ctx_dim, dtype=dtype)
        self.text_proxy_tokens_photo = nn.ParameterList(
            [nn.Parameter(text_proxy_token_photo.clone()) for _ in range(self.cross_layer_photo)]
        )
        visual_proxy_token_photo = torch.randn(1, v_dim, dtype=dtype)
        self.visual_proxy_tokens_photo = nn.ParameterList(
            [nn.Parameter(visual_proxy_token_photo.clone()) for _ in range(self.cross_layer_photo, self.prompt_depth_photo)]
        )

        ######## SKETCH branch params ########
        self.ctx_sketch = nn.Parameter(ctx_vectors_sketch.clone())
        self.cross_prompts_text_sketch = nn.ParameterList(
            [self.ctx_sketch] +
            [nn.Parameter(torch.empty(self.n_ctx_sketch, 512, dtype=dtype)) for _ in range(self.prompt_depth_sketch - 1)]
        )
        for single_para in self.cross_prompts_text_sketch[1:]:
            nn.init.normal_(single_para, std=0.02)

        self.cross_prompts_visual_sketch = nn.ParameterList(
            [nn.Parameter(visual_vectors_sketch.clone()) for _ in range(self.prompt_depth_sketch)]
        )

        self.text2visual_net_sketch = CrossPromptAttention(hidden_size=v_dim, encoder_hidden_size=ctx_dim, num_attention_heads=8)
        self.visual2text_net_sketch = CrossPromptAttention(hidden_size=ctx_dim, encoder_hidden_size=v_dim, num_attention_heads=8)
        self.photo_to_sketch_net = CrossPromptAttention(hidden_size=v_dim, encoder_hidden_size=v_dim, num_attention_heads=8)

        attn_pooling_text_sketch = AttentionPooling(hidden_size=ctx_dim, num_attention_heads=8)
        self.attn_pooling_text_nets_sketch = _get_clones(attn_pooling_text_sketch, self.cross_layer_sketch)
        attn_pooling_visual_sketch = AttentionPooling(hidden_size=v_dim, num_attention_heads=8)
        self.attn_pooling_visual_nets_sketch = _get_clones(attn_pooling_visual_sketch, self.prompt_depth_sketch - self.cross_layer_sketch)

        text_proxy_token_sketch = torch.randn(1, ctx_dim, dtype=dtype)
        self.text_proxy_tokens_sketch = nn.ParameterList(
            [nn.Parameter(text_proxy_token_sketch.clone()) for _ in range(self.cross_layer_sketch)]
        )
        visual_proxy_token_sketch = torch.randn(1, v_dim, dtype=dtype)
        self.visual_proxy_tokens_sketch = nn.ParameterList(
            [nn.Parameter(visual_proxy_token_sketch.clone()) for _ in range(self.cross_layer_sketch, self.prompt_depth_sketch)]
        )

        if self.prec_photo == "fp16":
            self.text2visual_net_photo = self.text2visual_net_photo.half()
            self.visual2text_net_photo = self.visual2text_net_photo.half()
            self.sketch2visual_net_photo = self.sketch2visual_net_photo.half()
            self.gate_alpha = nn.Parameter(self.gate_alpha.data.half())
            self.attn_pooling_text_nets_photo = self.attn_pooling_text_nets_photo.half()
            self.attn_pooling_visual_nets_photo = self.attn_pooling_visual_nets_photo.half()

        if self.prec_sketch == "fp16":
            self.text2visual_net_sketch = self.text2visual_net_sketch.half()
            self.visual2text_net_sketch = self.visual2text_net_sketch.half()
            self.photo_to_sketch_net = self.photo_to_sketch_net.half()
            self.attn_pooling_text_nets_sketch = self.attn_pooling_text_nets_sketch.half()
            self.attn_pooling_visual_nets_sketch = self.attn_pooling_visual_nets_sketch.half()

        # Backward compatibility aliases for existing code paths/tests.
        self.ctx = self.ctx_photo
        self.cross_prompts_text = self.cross_prompts_text_photo
        self.cross_prompts_visual = self.cross_prompts_visual_photo
        self.text2visual_net = self.text2visual_net_photo
        self.visual2text_net = self.visual2text_net_photo
        self.attn_pooling_text_nets = self.attn_pooling_text_nets_photo
        self.attn_pooling_visual_nets = self.attn_pooling_visual_nets_photo
        self.text_proxy_tokens = self.text_proxy_tokens_photo
        self.visual_proxy_tokens = self.visual_proxy_tokens_photo

        ######## Distillation Image Encoder ########
        self.ZS_image_encoder = self.clip_model_distill.visual

        # Cache động để lưu classnames của batch 
        self._cached_classnames = None
        self._cached_tokenized_prompts = None
        self._cached_fixed_embeddings = None
        self._cached_text_input_photo = None
        self._cached_text_input_sketch = None

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        return torch.cat([prefix, ctx, suffix], dim=1)

    def _prepare_dynamic_classnames(self, classnames, modality):
        """
        [CHỈNH SỬA LOAD DATA SKETCH_VLM]: 
        Bỏ triệt để GPT Classifier đọc JSON mapping rườm rà.
        Test Zero-Shot trên Sketchy chỉ cần gọi trực tiếp hàm encode_text tĩnh.
        """
        if modality == "photo":
            cross_prompts_text = self.cross_prompts_text_photo
        elif modality == "sketch":
            cross_prompts_text = self.cross_prompts_text_sketch
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if self._cached_classnames is None or classnames != self._cached_classnames:
            device = cross_prompts_text[0].device
            classnames_clean = [name.replace("_", " ") for name in classnames]
            if modality == "photo":
                prompt_prefix = self.ctx_init_text_photo if self.ctx_init_text_photo else " ".join(["X"] * self.n_ctx_photo)
            else:
                prompt_prefix = self.ctx_init_text_sketch if self.ctx_init_text_sketch else " ".join(["X"] * self.n_ctx_sketch)
            raw_prompts = [prompt_prefix + " " + name + "." for name in classnames_clean]

            from src.clip import clip as _clip
            tokenized_prompts = torch.cat([_clip.tokenize(p) for p in raw_prompts]).to(device)

            with torch.no_grad():
                fixed_embeddings = self.clip_model_distill.encode_text(tokenized_prompts)
                fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
                fixed_embeddings = fixed_embeddings.type(self.dtype)

            self._cached_classnames = classnames
            self._cached_tokenized_prompts = tokenized_prompts
            self._cached_fixed_embeddings = fixed_embeddings
            self._cached_text_input_photo = None
            self._cached_text_input_sketch = None
            
        text_input = self._cached_text_input_photo if modality == "photo" else self._cached_text_input_sketch
        if text_input is None:
            tokenized_prompts = self._cached_tokenized_prompts
            n_cls = len(classnames)
            n_ctx = self.n_ctx_photo if modality == "photo" else self.n_ctx_sketch
            with torch.no_grad():
                embedding = self.token_embedding(tokenized_prompts).type(self.dtype)

            token_prefix = embedding[:, :1, :]
            token_suffix = embedding[:, 1 + n_ctx:, :]

            ctx = cross_prompts_text[0]
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

            text_input = self.construct_prompts(ctx, token_prefix, token_suffix)
            if modality == "photo":
                self._cached_text_input_photo = text_input
            else:
                self._cached_text_input_sketch = text_input

        return text_input, self._cached_tokenized_prompts, self._cached_fixed_embeddings

    def get_early_text_proxies(self, modality):
        if modality == "photo":
            attn_pooling_text_nets = self.attn_pooling_text_nets_photo
            text_proxy_tokens = self.text_proxy_tokens_photo
            cross_prompts_text = self.cross_prompts_text_photo
            cross_layer = self.cross_layer_photo
        elif modality == "sketch":
            attn_pooling_text_nets = self.attn_pooling_text_nets_sketch
            text_proxy_tokens = self.text_proxy_tokens_sketch
            cross_prompts_text = self.cross_prompts_text_sketch
            cross_layer = self.cross_layer_sketch
        else:
            raise ValueError(f"Unknown modality: {modality}")
        proxy_text_tokens = []
        for i in range(cross_layer):
            text_proxy_token = attn_pooling_text_nets[i](
                token_query=text_proxy_tokens[i],
                sequence_key=cross_prompts_text[i],
                sequence_value=cross_prompts_text[i]
            )
            proxy_text_tokens.append(text_proxy_token)
        return torch.cat(proxy_text_tokens, dim=0).view(cross_layer, self.ctx_dim)

    def get_first_visual_prompt(self, modality):
        if modality == "photo":
            return self.cross_prompts_visual_photo[0]
        if modality == "sketch":
            return self.cross_prompts_visual_sketch[0]
        raise ValueError(f"Unknown modality: {modality}")

    def _build_external_shallow_visual_proxies_from_sketch(self, device, dtype, sketch_shallow_snapshot=None):
        """Build sketch shallow proxies as [cross_layer_photo, v_dim] for additive injection."""
        if self.cross_layer_sketch == 0:
            base = torch.zeros((1, self.v_dim), device=device, dtype=dtype)
        else:
            proxies = []
            source_visual_prompts = sketch_shallow_snapshot if sketch_shallow_snapshot is not None else self.cross_prompts_visual_sketch
            for i in range(self.cross_layer_sketch):
                # Summarize each shallow sketch layer into one proxy vector.
                proxies.append(source_visual_prompts[i].mean(dim=0, keepdim=True))
            base = torch.cat(proxies, dim=0).to(device=device, dtype=dtype)

        if base.shape[0] == self.cross_layer_photo:
            return base
        if base.shape[0] > self.cross_layer_photo:
            return base[:self.cross_layer_photo]

        repeat_factor = (self.cross_layer_photo + base.shape[0] - 1) // base.shape[0]
        return base.repeat(repeat_factor, 1)[:self.cross_layer_photo]

    def _build_branch_output(self, classnames, modality):
        text_input, tokenized_prompts, fixed_embeddings = self._prepare_dynamic_classnames(classnames, modality)

        if modality == "photo":
            cross_prompts_text = self.cross_prompts_text_photo
            cross_prompts_visual = self.cross_prompts_visual_photo
            prompt_depth = self.prompt_depth_photo
        elif modality == "sketch":
            cross_prompts_text = self.cross_prompts_text_sketch
            cross_prompts_visual = self.cross_prompts_visual_sketch
            prompt_depth = self.prompt_depth_sketch
        else:
            raise ValueError(f"Unknown modality: {modality}")

        cross_prompts_text_deeper = [cross_prompts_text[i] for i in range(1, len(cross_prompts_text))]
        cross_prompts_visual_deeper = [cross_prompts_visual[i] for i in range(1, len(cross_prompts_visual))]

        return text_input, tokenized_prompts, fixed_embeddings, cross_prompts_text_deeper, cross_prompts_visual_deeper

    def _apply_photo_to_sketch_deep_flow(self):
        """Update sketch deep visual prompts using the already-updated photo prompts."""
        for i in range(self.cross_layer_sketch, self.prompt_depth_sketch):
            photo_index = min(i, self.prompt_depth_photo - 1)
            photo_source = self.cross_prompts_visual_photo[photo_index]
            sketch_target = self.cross_prompts_visual_sketch[i]
            updated_sketch = self.photo_to_sketch_net(
                sketch_target,
                photo_source,
                photo_source,
            )
            self.cross_prompts_visual_sketch[i].data.copy_(updated_sketch)

    def _forward_single_modality(self, classnames, modality, sketch_shallow_snapshot=None):
        if modality == "photo":
            cross_prompts_text = self.cross_prompts_text_photo
            cross_prompts_visual = self.cross_prompts_visual_photo
            text2visual_net = self.text2visual_net_photo
            visual2text_net = self.visual2text_net_photo
            attn_pooling_visual_nets = self.attn_pooling_visual_nets_photo
            visual_proxy_tokens = self.visual_proxy_tokens_photo
            cross_layer = self.cross_layer_photo
            prompt_depth = self.prompt_depth_photo
        elif modality == "sketch":
            cross_prompts_text = self.cross_prompts_text_sketch
            cross_prompts_visual = self.cross_prompts_visual_sketch
            text2visual_net = self.text2visual_net_sketch
            visual2text_net = self.visual2text_net_sketch
            attn_pooling_visual_nets = self.attn_pooling_visual_nets_sketch
            visual_proxy_tokens = self.visual_proxy_tokens_sketch
            cross_layer = self.cross_layer_sketch
            prompt_depth = self.prompt_depth_sketch
        else:
            raise ValueError(f"Unknown modality: {modality}")

        text_input, tokenized_prompts, fixed_embeddings = self._prepare_dynamic_classnames(classnames, modality)

        ######## T->I mapping ########
        if modality == "photo":
            visual_prompts = torch.cat([cross_prompts_visual[i].unsqueeze(0) for i in range(cross_layer)], dim=0)
            proxy_text_prompts = self.get_early_text_proxies(modality).view(-1, self.ctx_dim)
            visual_prompts_flat = visual_prompts.view(-1, visual_prompts.shape[-1])

            # Preserve the baseline full text->visual update.
            updated_by_text = text2visual_net(
                visual_prompts_flat,
                proxy_text_prompts,
                proxy_text_prompts,
            )
            updated_by_text = updated_by_text.view(cross_layer, self.n_ctx_photo, self.v_dim)

            # Add sketch knowledge as a gated residual.
            external_shallow_visual_proxies = self._build_external_shallow_visual_proxies_from_sketch(
                device=visual_prompts.device,
                dtype=visual_prompts.dtype,
                sketch_shallow_snapshot=sketch_shallow_snapshot,
            )
            sketch_contribution = self.sketch2visual_net_photo(
                visual_prompts_flat,
                external_shallow_visual_proxies,
                external_shallow_visual_proxies,
            )
            sketch_contribution = sketch_contribution.view(cross_layer, self.n_ctx_photo, self.v_dim)

            updated_visual_prompts = updated_by_text + self.gate_alpha.to(dtype=visual_prompts.dtype) * sketch_contribution
            for i in range(cross_layer):
                cross_prompts_visual[i].data.copy_(updated_visual_prompts[i])
        else:
            visual_prompts = torch.cat([cross_prompts_visual[i].unsqueeze(0) for i in range(cross_layer)], dim=0)
            proxy_text_prompts = self.get_early_text_proxies(modality).view(-1, self.ctx_dim)

            visual_prompts_flat = visual_prompts.view(-1, visual_prompts.shape[-1])
            updated_by_text = text2visual_net(visual_prompts_flat, proxy_text_prompts, proxy_text_prompts)
            updated_visual_prompts = updated_by_text.view(cross_layer, -1, updated_by_text.shape[-1])

            for i in range(cross_layer):
                cross_prompts_visual[i].data.copy_(updated_visual_prompts[i])

        ######## I->T mapping ########
        text_prompts = torch.cat([cross_prompts_text[i].unsqueeze(0) for i in range(cross_layer, prompt_depth)], dim=0)
        visual_prompts = torch.cat([cross_prompts_visual[i].unsqueeze(0) for i in range(cross_layer, prompt_depth)], dim=0)
        proxy_visual_tokens = []
        for i in range(cross_layer, prompt_depth):
            visual_proxy_token = attn_pooling_visual_nets[i - cross_layer](
                token_query=visual_proxy_tokens[i - cross_layer],
                sequence_key=cross_prompts_visual[i],
                sequence_value=cross_prompts_visual[i]
            )
            proxy_visual_tokens.append(visual_proxy_token)
            proxy_visual_prompts = torch.cat(proxy_visual_tokens, dim=0)
        text_prompts = text_prompts.view(-1, text_prompts.shape[-1])
        proxy_visual_prompts = proxy_visual_prompts.view(-1, proxy_visual_prompts.shape[-1])
        updated_text_prompts = visual2text_net(text_prompts, proxy_visual_prompts, proxy_visual_prompts)
        updated_text_prompts = updated_text_prompts.view(prompt_depth - cross_layer, -1, updated_text_prompts.shape[-1])
        for i in range(cross_layer, prompt_depth):
            cross_prompts_text[i].data.copy_(updated_text_prompts[i - cross_layer])

        cross_prompts_text_deeper = [cross_prompts_text[i] for i in range(1, len(cross_prompts_text))]
        cross_prompts_visual_deeper = [cross_prompts_visual[i] for i in range(1, len(cross_prompts_visual))]

        return text_input, tokenized_prompts, fixed_embeddings, cross_prompts_text_deeper, cross_prompts_visual_deeper

    def forward_all(self, classnames):
        sketch_shallow_snapshot = [
            self.cross_prompts_visual_sketch[i].clone()
            for i in range(self.cross_layer_sketch)
        ]

        sketch_outputs = self._forward_single_modality(classnames, "sketch")
        photo_outputs = self._forward_single_modality(
            classnames,
            "photo",
            sketch_shallow_snapshot=sketch_shallow_snapshot,
        )
        self._apply_photo_to_sketch_deep_flow()
        sketch_outputs = self._build_branch_output(classnames, "sketch")

        return {
            "photo": photo_outputs,
            "sketch": sketch_outputs,
        }

    def forward(
        self,
        classnames,
        modality="photo",
    ):
        return self._forward_single_modality(classnames, modality)


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

