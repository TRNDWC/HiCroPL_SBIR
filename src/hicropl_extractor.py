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
    ):
        super().__init__()
        self.prompt_learner = prompt_learner
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.logit_scale = logit_scale
        self.dtype = dtype

    def forward(self, image, label=None):
        # 1. Gọi Prompt Learner
        text_input, first_visual_prompt, cross_prompts_text_deeper, cross_prompts_visual_deeper = self.prompt_learner()
        
        # 2. Trích xuất Đặc trưng (Có học)
        text_features_all = self.text_encoder(text_input, self.prompt_learner.tokenized_prompts, cross_prompts_text_deeper)
        image_features = self.image_encoder(image.type(self.dtype), first_visual_prompt, cross_prompts_visual_deeper)

        # 3. Phân tách Label cho Text Features
        if label is not None:
            text_features_batch = text_features_all[label]
        else:
            text_features_batch = text_features_all

        return {
            "image_features": image_features,
            "text_features":  text_features_batch,
            "text_features_all": text_features_all,
            "logit_scale": self.logit_scale.exp()
        }

