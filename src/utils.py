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


def load_class_descriptions(path):
    """Read a class-description JSON -> (dict classname->text, 8-hex fingerprint).

    Accepts BOTH shapes seen in gpt_file/:
      * flat        {"airplane": "Extends two wings...", ...}
      * with meta   {"meta": {...}, "descriptions": {"airplane": "...", ...}}
    The nested form is detected by the presence of a 'descriptions' key holding a
    dict; anything else is read as the flat form.

    The fingerprint covers ONLY the classname->text mapping, never a surrounding
    `meta` block: that block carries the generation run's metrics, which move
    every time the generator is re-run even when the text is byte-identical, and
    a fingerprint that changes while the training input did not is worse than no
    fingerprint. Serialized with sort_keys so key order never affects it.
    """
    import json
    import hashlib

    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(payload).__name__}")
    inner = payload.get('descriptions')
    descriptions = inner if isinstance(inner, dict) else payload
    if not descriptions:
        raise ValueError(f"{path}: no class descriptions found (empty mapping)")
    bad = [k for k, v in descriptions.items() if not isinstance(v, str)]
    if bad:
        raise ValueError(
            f"{path}: {len(bad)} entries are not strings, e.g. {bad[:5]}. Expected a flat "
            f"{{classname: text}} mapping, or {{'descriptions': {{classname: text}}}}."
        )
    blob = json.dumps(descriptions, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return descriptions, hashlib.sha256(blob).hexdigest()[:8]
