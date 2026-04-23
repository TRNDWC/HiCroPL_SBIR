import torch
from src.clip import clip

def load_clip_to_cpu(opts, zero_shot_model=False):
    """
    Load CLIP model to CPU and rebuild with design details (HiCroPL/MaPLe style).
    
    Args:
        opts: Configuration object with backbone, prompt_depth, n_ctx, etc.
        zero_shot_model: If True, uses 'IVLP' trainer with zero prompt depth.
    """
    backbone_name = opts.backbone
    
    if not zero_shot_model:
        vision_depth = opts.prompt_depth if opts.vision_depth < 0 else opts.vision_depth
        language_depth = opts.prompt_depth if opts.language_depth < 0 else opts.language_depth
        # Visual encoder receives concatenated prompts from CrossModal and CrossVisual learners.
        # Default to 2 * n_ctx unless user explicitly overrides vision_ctx.
        vision_ctx = (2 * opts.n_ctx) if opts.vision_ctx < 0 else opts.vision_ctx
        language_ctx = opts.n_ctx if opts.language_ctx < 0 else opts.language_ctx
        
        design_details = {
            "trainer": opts.clip_trainer,
            "vision_depth": vision_depth,
            "language_depth": language_depth,
            "vision_ctx": vision_ctx,
            "language_ctx": language_ctx,
        }
    else:
        # Return base CLIP model (IVLP with depth 0) for generating frozen VL features
        design_details = {
            "trainer": 'IVLP',
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
    
    # clip.load already handles build_model internally if design_details is provided.
    # We use "cpu" to match the original HiCroPL function name, though it can be moved to GPU later.
    model, _ = clip.load(backbone_name, device="cpu", design_details=design_details)
    return model

def load_clip_to_cpu_teacher(opts):
    """
    Load the frozen teacher (distill) CLIP model. 
    Matches the pattern in HiCroPL.
    """
    return load_clip_to_cpu(opts, zero_shot_model=True)
