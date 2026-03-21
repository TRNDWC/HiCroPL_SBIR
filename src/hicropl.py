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
    def __init__(self, cfg, clip_model, clip_model_distill=None):
        super().__init__()
        
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

        self.n_ctx = n_ctx
        self.ctx_init_text = ctx_init
        self.dtype = dtype
        self.token_embedding = clip_model.token_embedding
        self.clip_model = clip_model 
        
        # [SỬA ĐỔI LOAD DATA]: Trong framework SBIR, không có teacher gốc mà dùng chung distill branch
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
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

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
        
        self.text_proxy_tokens = nn.ParameterList([nn.Parameter(torch.randn(1, ctx_dim, dtype=dtype)) for _ in range(self.cross_layer)])
        self.visual_proxy_tokens = nn.ParameterList([nn.Parameter(torch.randn(1, v_dim, dtype=dtype)) for _ in range(self.cross_layer, self.prompt_depth)])
        
        if prec == "fp16":
            self.attn_pooling_text_nets, self.attn_pooling_visual_nets = self.attn_pooling_text_nets.half(), self.attn_pooling_visual_nets.half()

        ######## Distillation Image Encoder ########
        self.ZS_image_encoder = self.clip_model_distill.visual

        # Cache động để lưu classnames của batch 
        self._cached_classnames = None
        self._cached_text_input = None
        self._cached_tokenized_prompts = None
        self._cached_fixed_embeddings = None

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        return torch.cat([prefix, ctx, suffix], dim=1)

    def _prepare_dynamic_classnames(self, classnames):
        """
        [CHỈNH SỬA LOAD DATA SKETCH_VLM]: 
        Bỏ triệt để GPT Classifier đọc JSON mapping rườm rà.
        Test Zero-Shot trên Sketchy chỉ cần gọi trực tiếp hàm encode_text tĩnh.
        """
        if self._cached_classnames is not None and classnames == self._cached_classnames:
            return self._cached_text_input, self._cached_tokenized_prompts, self._cached_fixed_embeddings
            
        n_cls = len(classnames)
        device = self.cross_prompts_text[0].device
        
        classnames_clean = [name.replace("_", " ") for name in classnames]
        prompt_prefix = self.ctx_init_text if self.ctx_init_text else " ".join(["X"] * self.n_ctx)
        raw_prompts = [prompt_prefix + " " + name + "." for name in classnames_clean]
        
        from src.clip import clip as _clip
        tokenized_prompts = torch.cat([_clip.tokenize(p) for p in raw_prompts]).to(device)
        
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)  
            
        token_prefix = embedding[:, :1, :]
        token_suffix = embedding[:, 1 + self.n_ctx:, :]
        
        ctx = self.cross_prompts_text[0]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)
            
        text_input = self.construct_prompts(ctx, token_prefix, token_suffix)

        # [SỬA ĐỔI LOAD DATA]: Băm text thành Fixed Embeddings bằng CLIP model gốc (thay vì read folder/json gpt_clip_classifier)
        with torch.no_grad():
            fixed_embeddings = self.clip_model_distill.encode_text(tokenized_prompts)
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            fixed_embeddings = fixed_embeddings.type(self.dtype)
        
        self._cached_classnames = classnames
        self._cached_text_input = text_input
        self._cached_tokenized_prompts = tokenized_prompts
        self._cached_fixed_embeddings = fixed_embeddings

        return text_input, tokenized_prompts, fixed_embeddings

    def forward(self, classnames):
        text_input, tokenized_prompts, fixed_embeddings = self._prepare_dynamic_classnames(classnames)

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
        
        return text_input, tokenized_prompts, fixed_embeddings, cross_prompts_text_deeper, cross_prompts_visual_deeper


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

