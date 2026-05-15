import torch
import torch.nn as nn
import torch.nn.functional as F


class JigsawNet(nn.Module):
    """Simple per-patch jigsaw predictor.

    - Expects `patches_feat` shaped (B, P, D)
    - Optional `context` shaped (B, D_ctx) will be concatenated to each patch.
    - Returns logits shaped (B, P, P) where logits[b, j, k] is score that
      shuffled patch at position j comes from original position k.
    """
    def __init__(self, num_patches: int, feat_dim: int, hidden: int = 512):
        super().__init__()
        self.num_patches = num_patches
        self.feat_dim = feat_dim
        self.hidden = hidden

        self.fc1 = nn.Linear(feat_dim * 2, hidden)
        self.fc2 = nn.Linear(hidden, num_patches)

    def forward(self, patches_feat: torch.Tensor, context: torch.Tensor = None):
        # patches_feat: (B, P, D)
        B, P, D = patches_feat.shape
        if context is None:
            # zero context
            context = torch.zeros(B, D, device=patches_feat.device, dtype=patches_feat.dtype)

        # broadcast context to per-patch
        ctx = context.unsqueeze(1).expand(-1, P, -1)  # (B, P, D)
        x = torch.cat([patches_feat, ctx], dim=-1)  # (B, P, 2D)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  # (B, P, P)
        return logits
