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
    def __init__(self, cfg, classnames, clip_model, clip_model_distill=None):
        super().__init__()
        
        n_cls = len(classnames)
        self.prompt_depth = getattr(cfg, 'prompt_depth', 9)
        self.cross_layer = getattr(cfg, 'cross_layer', 4)
        n_ctx = getattr(cfg, 'n_ctx', 4)
        ctx_init = getattr(cfg, 'ctx_init', "a photo of a")
        self.dataset_name = getattr(cfg, 'dataset', 'sketchy')
        prec = getattr(cfg, 'prec', "fp32")
        
        assert self.prompt_depth >= 1, "Language prompt depth should be >=1"
        
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim if hasattr(clip_model.visual, 'output_dim') else clip_model.visual.conv1.weight.shape[0]
        v_dim = 768

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.dtype = dtype
        self.token_embedding = clip_model.token_embedding
        self.clip_model = clip_model 
        
        # Store distill model for zero-shot image encoder
        self.clip_model_distill = clip_model_distill if clip_model_distill is not None else clip_model

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
        visual_vectors = torch.empty(n_ctx, v_dim, dtype=dtype)
        nn.init.normal_(visual_vectors, std=0.02)
        cross_prompts_visual = nn.ParameterList([nn.Parameter(visual_vectors) for _ in range(self.prompt_depth)])
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

        ######## Distillation Image Encoder ########
        self.ZS_image_encoder = self.clip_model_distill.visual

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
        device = self.cross_prompts_text[0].device
        ctx = self.cross_prompts_text[0]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        # Construct text input prompts
        text_input = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        ######## T->I mapping ########
        visual_prompts = torch.cat([self.cross_prompts_visual[i].unsqueeze(0) for i in range(self.cross_layer)], dim=0)  
        text_prompts = torch.cat([self.cross_prompts_text[i].unsqueeze(0) for i in range(self.cross_layer)], dim=0)  
        proxy_text_tokens = []
        for i in range(self.cross_layer):
            text_proxy_token = self.attn_pooling_text_nets[i](
                token_query=self.text_proxy_tokens[i],  
                sequence_key=self.cross_prompts_text[i],  
                sequence_value=self.cross_prompts_text[i]  
            )
            proxy_text_tokens.append(text_proxy_token)
        proxy_text_prompts = torch.cat(proxy_text_tokens, dim=0)  
        visual_prompts = visual_prompts.view(-1, visual_prompts.shape[-1])  
        proxy_text_prompts = proxy_text_prompts.view(-1, proxy_text_prompts.shape[-1])  
        updated_visual_prompts = self.text2visual_net(visual_prompts, proxy_text_prompts, proxy_text_prompts)  
        updated_visual_prompts = updated_visual_prompts.view(self.cross_layer, -1, updated_visual_prompts.shape[-1])  
        for i in range(self.cross_layer):
            self.cross_prompts_visual[i].data.copy_(updated_visual_prompts[i])

        ######## I->T mapping ########
        text_prompts = torch.cat([self.cross_prompts_text[i].unsqueeze(0) for i in range(self.cross_layer, self.prompt_depth)], dim=0)  
        visual_prompts = torch.cat([self.cross_prompts_visual[i].unsqueeze(0) for i in range(self.cross_layer, self.prompt_depth)], dim=0)  
        proxy_visual_tokens = []
        for i in range(self.cross_layer, self.prompt_depth):
            visual_proxy_token = self.attn_pooling_visual_nets[i - self.cross_layer](
                token_query=self.visual_proxy_tokens[i - self.cross_layer],  
                sequence_key=self.cross_prompts_visual[i],  
                sequence_value=self.cross_prompts_visual[i]  
            )
            proxy_visual_tokens.append(visual_proxy_token)
            proxy_visual_prompts = torch.cat(proxy_visual_tokens, dim=0)  
        text_prompts = text_prompts.view(-1, text_prompts.shape[-1])  
        proxy_visual_prompts = proxy_visual_prompts.view(-1, proxy_visual_prompts.shape[-1])  
        updated_text_prompts = self.visual2text_net(text_prompts, proxy_visual_prompts, proxy_visual_prompts)  
        updated_text_prompts = updated_text_prompts.view(self.prompt_depth - self.cross_layer, -1, updated_text_prompts.shape[-1])  
        for i in range(self.cross_layer, self.prompt_depth):
            self.cross_prompts_text[i].data.copy_(updated_text_prompts[i - self.cross_layer])

        cross_prompts_text_deeper = [self.cross_prompts_text[i] for i in range(1, len(self.cross_prompts_text))]
        cross_prompts_visual_deeper = [self.cross_prompts_visual[i] for i in range(1, len(self.cross_prompts_visual))]
        
        return text_input, self.cross_prompts_visual[0], cross_prompts_text_deeper, cross_prompts_visual_deeper


class VisualVisualPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model_photo, clip_model_sketch):
        super().__init__()

        self.prompt_depth = getattr(cfg, 'prompt_depth', 9)
        self.cross_layer = getattr(cfg, 'cross_layer', 4)
        n_ctx = getattr(cfg, 'n_ctx', 4)
        prec = getattr(cfg, 'prec', 'fp32')

        assert self.prompt_depth >= 1

        dtype = clip_model_photo.dtype
        # photo đóng vai text → dùng conv1 output dim làm "ctx_dim"
        p_dim = clip_model_photo.visual.conv1.weight.shape[0]   # 768, analog ctx_dim
        s_dim = clip_model_sketch.visual.conv1.weight.shape[0]  # 768, analog v_dim
        assert p_dim == s_dim, "Both branches must have same embedding dimension"

        self.dtype = dtype
        self.n_ctx = n_ctx

        ######## photo prompt initialization (analog: text initialization) ########
        # Text gốc dùng token_embedding("a photo of a") để có semantic prior
        # Photo analog: dùng conv1 patch embedding của ảnh mẫu để có visual prior
        # Nếu không có sample, fallback về random như visual gốc
        photo_vectors = torch.empty(n_ctx, p_dim, dtype=dtype)
        nn.init.normal_(photo_vectors, std=0.02)
        
        # Layer 0: learnable (analog self.ctx trong gốc)
        self.ctx_photo = nn.Parameter(photo_vectors)
        # Deeper layers: random init như gốc
        cross_prompts_photo = nn.ParameterList(
            [self.ctx_photo] + 
            [nn.Parameter(torch.empty(n_ctx, p_dim, dtype=dtype)) 
             for _ in range(self.prompt_depth - 1)]
        )
        for single_para in cross_prompts_photo[1:]:
            nn.init.normal_(single_para, std=0.02)
        self.cross_prompts_photo = cross_prompts_photo
        ######## photo prompt initialization end ########

        ######## sketch prompt initialization (analog: visual initialization) ########
        sketch_vectors = torch.empty(n_ctx, s_dim, dtype=dtype)
        nn.init.normal_(sketch_vectors, std=0.02)
        cross_prompts_sketch = nn.ParameterList(
            [nn.Parameter(sketch_vectors.clone()) 
             for _ in range(self.prompt_depth)]
        )
        self.cross_prompts_sketch = cross_prompts_sketch
        ######## sketch prompt initialization end ########

        ######## knowledge mapper: photo2sketch and sketch2photo ########
        # Analog: text2visual_net và visual2text_net
        # Vì p_dim == s_dim = 768, hidden_size = encoder_hidden_size = 768
        self.photo2sketch_net = CrossPromptAttention(
            hidden_size=s_dim, encoder_hidden_size=p_dim, num_attention_heads=8
        )
        self.sketch2photo_net = CrossPromptAttention(
            hidden_size=p_dim, encoder_hidden_size=s_dim, num_attention_heads=8
        )
        if prec == 'fp16':
            self.photo2sketch_net = self.photo2sketch_net.half()
            self.sketch2photo_net = self.sketch2photo_net.half()

        # LKP: photo pools (analog: attn_pooling_text_nets, cross_layer cái)
        attn_pooling_photo = AttentionPooling(hidden_size=p_dim, num_attention_heads=8)
        self.attn_pooling_photo_nets = _get_clones(attn_pooling_photo, self.cross_layer)

        # LKP: sketch pools (analog: attn_pooling_visual_nets, prompt_depth - cross_layer cái)
        attn_pooling_sketch = AttentionPooling(hidden_size=s_dim, num_attention_heads=8)
        self.attn_pooling_sketch_nets = _get_clones(
            attn_pooling_sketch, self.prompt_depth - self.cross_layer
        )

        # Proxy tokens: photo proxy cho shallow (analog: text_proxy_tokens)
        photo_proxy_token = torch.randn(1, p_dim, dtype=dtype)
        self.photo_proxy_tokens = nn.ParameterList(
            [nn.Parameter(photo_proxy_token.clone()) for _ in range(self.cross_layer)]
        )

        # Proxy tokens: sketch proxy cho deep (analog: visual_proxy_tokens)
        sketch_proxy_token = torch.randn(1, s_dim, dtype=dtype)
        self.sketch_proxy_tokens = nn.ParameterList(
            [nn.Parameter(sketch_proxy_token.clone()) 
             for _ in range(self.cross_layer, self.prompt_depth)]
        )

        if prec == 'fp16':
            self.attn_pooling_photo_nets = self.attn_pooling_photo_nets.half()
            self.attn_pooling_sketch_nets = self.attn_pooling_sketch_nets.half()
        ######## knowledge mapper end ########

    def forward(self):
        ######## P->S mapping (analog: T->I mapping) ########
        # Photo guides sketch ở shallow layers [0..cross_layer]
        sketch_prompts = torch.cat(
            [self.cross_prompts_sketch[i].unsqueeze(0) for i in range(self.cross_layer)], dim=0
        )
        # LKP: compress photo prompts thành proxy
        proxy_photo_tokens = []
        for i in range(self.cross_layer):
            photo_proxy = self.attn_pooling_photo_nets[i](
                token_query=self.photo_proxy_tokens[i],
                sequence_key=self.cross_prompts_photo[i],
                sequence_value=self.cross_prompts_photo[i]
            )
            proxy_photo_tokens.append(photo_proxy)
        proxy_photo_prompts = torch.cat(proxy_photo_tokens, dim=0)

        sketch_prompts_flat = sketch_prompts.view(-1, sketch_prompts.shape[-1])
        proxy_photo_flat = proxy_photo_prompts.view(-1, proxy_photo_prompts.shape[-1])

        updated_sketch_prompts = self.photo2sketch_net(
            sketch_prompts_flat, proxy_photo_flat, proxy_photo_flat
        )
        updated_sketch_prompts = updated_sketch_prompts.view(
            self.cross_layer, -1, updated_sketch_prompts.shape[-1]
        )
        for i in range(self.cross_layer):
            self.cross_prompts_sketch[i].data.copy_(updated_sketch_prompts[i])
        ######## P->S mapping end ########

        ######## S->P mapping (analog: I->T mapping) ########
        # Sketch guides photo ở deep layers [cross_layer..prompt_depth]
        photo_prompts = torch.cat(
            [self.cross_prompts_photo[i].unsqueeze(0) 
             for i in range(self.cross_layer, self.prompt_depth)], dim=0
        )
        # LKP: compress sketch prompts thành proxy
        proxy_sketch_tokens = []
        for i in range(self.cross_layer, self.prompt_depth):
            sketch_proxy = self.attn_pooling_sketch_nets[i - self.cross_layer](
                token_query=self.sketch_proxy_tokens[i - self.cross_layer],
                sequence_key=self.cross_prompts_sketch[i],
                sequence_value=self.cross_prompts_sketch[i]
            )
            proxy_sketch_tokens.append(sketch_proxy)
            proxy_sketch_prompts = torch.cat(proxy_sketch_tokens, dim=0)

        photo_prompts_flat = photo_prompts.view(-1, photo_prompts.shape[-1])
        proxy_sketch_flat = proxy_sketch_prompts.view(-1, proxy_sketch_prompts.shape[-1])

        updated_photo_prompts = self.sketch2photo_net(
            photo_prompts_flat, proxy_sketch_flat, proxy_sketch_flat
        )
        updated_photo_prompts = updated_photo_prompts.view(
            self.prompt_depth - self.cross_layer, -1, updated_photo_prompts.shape[-1]
        )
        for i in range(self.cross_layer, self.prompt_depth):
            self.cross_prompts_photo[i].data.copy_(updated_photo_prompts[i - self.cross_layer])
        ######## S->P mapping end ########

        # Extract deeper prompts (analog: cross_prompts_text_deeper, cross_prompts_visual_deeper)
        cross_prompts_photo_deeper = [
            self.cross_prompts_photo[i] for i in range(1, len(self.cross_prompts_photo))
        ]
        cross_prompts_sketch_deeper = [
            self.cross_prompts_sketch[i] for i in range(1, len(self.cross_prompts_sketch))
        ]

        # Returns analog: (text_input, visual_ctx[0], text_deeper, visual_deeper)
        # Ở đây không có text_input vì đây là visual-visual
        # photo[0] = shallow photo prompt, sketch[0] = shallow sketch prompt
        return (
            self.cross_prompts_photo[0],   # analog: visual_ctx (first layer prompt)
            self.cross_prompts_sketch[0],  # analog: visual_ctx cho branch kia
            cross_prompts_photo_deeper,    # analog: cross_prompts_text_deeper
            cross_prompts_sketch_deeper    # analog: cross_prompts_visual_deeper
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
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.tokenized_prompts.shape[0], -1, -1)
        text_input = self.construct_prompts(ctx, self.token_prefix, self.token_suffix, label=label)
        return text_input, []  # shallow-only: no deep injection


class BranchPromptAdapter(nn.Module):
    """Adapter that exposes a unified prompt-learner interface for a branch.

    It composes a shared VisualVisualPromptLearner (for both visuals) and a
    SimpleTextPromptLearner for the branch's text prompts, and returns the
    4-tuple expected by HiCroPLFeatureExtractor.forward().
    """

    def __init__(self, visual_learner: VisualVisualPromptLearner, text_learner: SimpleTextPromptLearner, branch: str):
        super().__init__()
        assert branch in ("photo", "sketch")
        self.visual_learner = visual_learner
        self.text_learner = text_learner
        self.branch = branch
        # Proxy token buffers from the text learner so older callers (e.g.
        # `HiCroPLFeatureExtractor`) that access `prompt_learner.tokenized_prompts`
        # continue to work with the adapter.
        if hasattr(text_learner, 'tokenized_prompts'):
            self.register_buffer('tokenized_prompts', text_learner.tokenized_prompts)
        if hasattr(text_learner, 'token_prefix'):
            self.register_buffer('token_prefix', text_learner.token_prefix)
        if hasattr(text_learner, 'token_suffix'):
            self.register_buffer('token_suffix', text_learner.token_suffix)

    def forward(self, label=None):
        # run visual-visual learner to update both visual prompt sets
        vis1_shallow, vis2_shallow, vis1_deeper, vis2_deeper = self.visual_learner()

        # run text-only learner for this branch
        text_input, cross_prompts_text_deeper = self.text_learner(label=label)

        if self.branch == 'photo':
            first_visual_prompt = vis2_shallow
            cross_prompts_visual_deeper = vis2_deeper
        else:
            first_visual_prompt = vis1_shallow
            cross_prompts_visual_deeper = vis1_deeper

        return text_input, first_visual_prompt, cross_prompts_text_deeper, cross_prompts_visual_deeper

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

    