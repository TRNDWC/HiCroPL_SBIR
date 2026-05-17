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


class TextEncoder(nn.Module):
    """CLIP text encoder wrapper with deep prompt injection support."""
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
        combined = [x, cross_prompts_text_deeper]
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Take features from the EOT embedding
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class VisualVisualPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model_photo, clip_model_sketch):
        super().__init__()

        self.prompt_depth = getattr(cfg, 'prompt_depth', 9)
        self.cross_layer = getattr(cfg, 'cross_layer', 4)
        n_ctx = getattr(cfg, 'n_ctx', 4)
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
        # No bidirectional mapping: prompts are learned independently per modality

    def forward(self):
        # No bidirectional exchange: return learnable prompts directly

        # Extract deeper prompts (analog: cross_prompts_text_deeper, cross_prompts_visual_deeper)
        cross_prompts_photo_deeper = [
            self.cross_prompts_photo[i] for i in range(1, len(self.cross_prompts_photo))
        ]
        cross_prompts_sketch_deeper = [
            self.cross_prompts_sketch[i] for i in range(1, len(self.cross_prompts_sketch))
        ]

        # Returns analog: (text_input, visual_ctx[0], text_deeper, visual_deeper)
        # Ở đây không có text_input vì đây là visual-visual
        # sketch[0] = shallow sketch prompt, photo[0] = shallow photo prompt
        return (
            self.cross_prompts_sketch[0],  # analog: visual_ctx cho sketch branch
            self.cross_prompts_photo[0],   # analog: visual_ctx cho photo branch
            cross_prompts_sketch_deeper,   # analog: cross_prompts_text_deeper
            cross_prompts_photo_deeper     # analog: cross_prompts_visual_deeper
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

    