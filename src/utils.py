import torch
from src.clip import clip

def load_clip_to_cpu(opts):
    """
    Load CLIP model to CPU and rebuild with design details (HiCroPL/MaPLe style).

    Args:
        opts: Configuration object with backbone, vision_depth, text_depth, n_ctx, etc.
    """
    backbone_name = opts.backbone

    # vision_depth/text_depth each drive both ends independently: the number of
    # prompt tensors VisualVisualPromptLearner/SimpleTextPromptLearner create for
    # that branch, and the number of CLIP resblocks that consume them.
    design_details = {
        "trainer": opts.clip_trainer,
        "vision_depth": opts.vision_depth,
        "language_depth": opts.text_depth,
        "vision_ctx": opts.n_ctx,
        "language_ctx": opts.n_ctx,
    }

    # clip.load already handles build_model internally if design_details is provided.
    # We use "cpu" to match the original HiCroPL function name, though it can be moved to GPU later.
    model, _ = clip.load(backbone_name, device="cpu", design_details=design_details)
    return model
