import torch
from src.clip import clip

def load_clip_to_cpu(opts, zero_shot_model=False):
    """Load CLIP and build with cross-domain (XDom) deep-prompt routing on the visual side.

    With ``trainer='XDom'`` and ``vision_depth>0``, the visual Transformer uses
    ``ResidualAttentionBlock_XDom`` for the first ``vision_depth`` layers, which lets
    layer-i deep prompts swap into the trailing ``n_ctx`` slots of the patch stream.
    The text Transformer always stays vanilla.

    Args:
        opts: Config; reads ``backbone``, ``prompt_depth``, ``n_ctx``.
        zero_shot_model: If True, returns a vanilla CLIP (no deep-prompt blocks)
            suitable as a frozen teacher for distillation.
    """
    backbone_name = opts.backbone

    if zero_shot_model:
        design_details = {
            "trainer": "IVLP",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
    else:
        design_details = {
            "trainer": "XDom",
            "vision_depth": int(getattr(opts, "prompt_depth", 1)),
            "language_depth": 0,
            "vision_ctx": int(getattr(opts, "n_ctx", 3)),
            "language_ctx": 0,
        }

    model, _ = clip.load(backbone_name, device="cpu", design_details=design_details)
    return model

def load_clip_to_cpu_teacher(opts):
    """
    Load the frozen teacher (distill) CLIP model. 
    Matches the pattern in HiCroPL.
    """
    return load_clip_to_cpu(opts, zero_shot_model=True)
