import torch
from src.clip import clip

def load_clip_to_cpu(opts):
    """
    Load CLIP model to CPU and rebuild with design details (HiCroPL/MaPLe style).

    Args:
        opts: Configuration object with backbone, prompt_depth, n_ctx, etc.
    """
    backbone_name = opts.backbone

    vision_depth = opts.prompt_depth if opts.vision_depth < 0 else opts.vision_depth
    language_depth = opts.prompt_depth if opts.language_depth < 0 else opts.language_depth
    vision_ctx = opts.n_ctx if opts.vision_ctx < 0 else opts.vision_ctx
    language_ctx = opts.n_ctx if opts.language_ctx < 0 else opts.language_ctx

    design_details = {
        "trainer": opts.clip_trainer,
        "vision_depth": vision_depth,
        "language_depth": language_depth,
        "vision_ctx": vision_ctx,
        "language_ctx": language_ctx,
    }

    # clip.load already handles build_model internally if design_details is provided.
    # We use "cpu" to match the original HiCroPL function name, though it can be moved to GPU later.
    model, _ = clip.load(backbone_name, device="cpu", design_details=design_details)
    return model
