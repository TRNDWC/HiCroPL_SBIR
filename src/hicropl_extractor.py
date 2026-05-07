import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import copy

from src.clip.model import QuickGELU

# ==============================================================================
# LỚP WRAPPER ĐƯỢC CẬP NHẬT TƯƠNG THÍCH SKETCH_VLM ĐẦU VÀO ĐỘNG
# ==============================================================================
class HiCroPLFeatureExtractor(nn.Module):
    def __init__(
        self,
        prompt_learner,
        text_encoder,
        image_encoder,
        logit_scale,
        dtype,
        text_distill_tokenized_prompts=None,
    ):
        super().__init__()
        self.prompt_learner = prompt_learner
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.logit_scale = logit_scale
        self.dtype = dtype
        if text_distill_tokenized_prompts is not None:
            self.register_buffer("text_distill_tokenized_prompts", text_distill_tokenized_prompts)
        else:
            self.text_distill_tokenized_prompts = None

    def encode_text_distill(self):
        tokenized_prompts = (
            self.text_distill_tokenized_prompts
            if self.text_distill_tokenized_prompts is not None
            else self.prompt_learner.tokenized_prompts
        )
        tokenized_prompts = tokenized_prompts.to(next(self.prompt_learner.clip_model_distill.parameters()).device)
        text_features = self.prompt_learner.clip_model_distill.encode_text(tokenized_prompts)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def forward(self, image, label=None, image_distill=None):
        # 1. Gọi Prompt Learner (classnames đã được khởi tạo ở __init__)
        text_input, first_visual_prompt, cross_prompts_text_deeper, cross_prompts_visual_deeper = self.prompt_learner()
        
        # 2. Compute GPT text distill embeddings with trainable distill LayerNorm.
        text_features_fixed_all = self.encode_text_distill()
        
        # 3. Đặc trưng Zero-Shot Fixed Image
        image_distill = image if image_distill is None else image_distill
        with torch.no_grad():
            image_features_fixed = self.prompt_learner.ZS_image_encoder(image_distill.type(self.dtype))
            image_features_fixed = image_features_fixed / image_features_fixed.norm(dim=-1, keepdim=True)

        # 4. Trích xuất Đặc trưng (Có học)
        text_features_all = self.text_encoder(text_input, self.prompt_learner.tokenized_prompts, cross_prompts_text_deeper)
        image_features = self.image_encoder(image.type(self.dtype), first_visual_prompt, cross_prompts_visual_deeper)

        # 4. Chuẩn hóa & Residual Mix
        image_features_norm1 = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_prenorm = image_features_norm1 + image_features_fixed  # chưa normalize cuối -> dành cho Adapter
        image_features_final = image_features_prenorm / image_features_prenorm.norm(dim=-1, keepdim=True)
        
        text_features_norm1 = text_features_all / text_features_all.norm(dim=-1, keepdim=True)
        text_features_prenorm_all = text_features_norm1 + text_features_fixed_all  # chưa normalize cuối -> dành cho Adapter
        text_features_final_all = text_features_prenorm_all / text_features_prenorm_all.norm(dim=-1, keepdim=True)

        # 5. Phân tách Label
        if label is not None:
            text_features_batch          = text_features_final_all[label]
            text_features_prenorm_batch  = text_features_prenorm_all[label]
            text_features_fixed_batch    = text_features_fixed_all[label]
        else:
            text_features_batch          = text_features_final_all
            text_features_prenorm_batch  = text_features_prenorm_all
            text_features_fixed_batch    = text_features_fixed_all

        return {
            "image_features":         image_features_final,
            "image_features_prenorm": image_features_prenorm,
            "image_features_fixed":   image_features_fixed,
            "text_features":          text_features_batch,
            "text_features_prenorm":  text_features_prenorm_batch,
            "text_features_fixed":    text_features_fixed_batch,
            "text_features_all":      text_features_final_all,
            "logit_scale":            self.logit_scale.exp()
        }
