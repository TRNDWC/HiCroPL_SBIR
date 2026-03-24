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
    def __init__(self, prompt_learner, text_encoder, image_encoder, logit_scale, dtype):
        super().__init__()
        self.prompt_learner = prompt_learner
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.logit_scale = logit_scale
        self.dtype = dtype

    def forward(self, image, classnames, label=None):
        # 1. Gọi Prompt Learner 
        text_input, tokenized_prompts, text_features_fixed_all, cross_prompts_text_deeper, cross_prompts_visual_deeper = self.prompt_learner(classnames)
        
        # 2. Đặc trưng Zero-Shot Fixed Image
        with torch.no_grad():
            image_features_fixed = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
            image_features_fixed = image_features_fixed / image_features_fixed.norm(dim=-1, keepdim=True)

        # 3. Trích xuất Đặc trưng (Có học)
        text_features_all = self.text_encoder(text_input, tokenized_prompts, cross_prompts_text_deeper)
        image_features = self.image_encoder(image.type(self.dtype), self.prompt_learner.cross_prompts_visual[0], cross_prompts_visual_deeper)

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
