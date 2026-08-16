"""Read-only diagnostic feature export for HiCroPL-SBIR.

Loads a checkpoint (or, with --frozen, skips the checkpoint entirely and uses
raw pretrained CLIP with a fixed template / no prompt learning), runs
inference only (torch.no_grad(), model.eval(), no augmentation), and dumps
every raw feature tensor to a single .npz file. Computes NO metric of any
kind (no mAP, no precision) -- export only.

Usage:
    python -m tools.diag_export --ckpt <path.ckpt> --data_dir <path> \
        --dataset sketchy_ext --out <path.npz>
    python -m tools.diag_export --frozen --data_dir <path> \
        --dataset sketchy_ext --out <path.npz>

IMPORTANT LIMITATION (see project audit, Q7): checkpoints produced by this
repo's ModelCheckpoint never save hyperparameters (HiCroPL_SBIR.__init__
never calls self.save_hyperparameters()). The architecture
(prompt_depth / n_ctx / cross_layer / n_proxy / --exchange_free_source /
--sketch_self_refine_ln) is therefore INFERRED here from the checkpoint's own
state_dict tensor shapes and key presence -- not read from any saved config.
This is reliable because it is exactly the information load_state_dict(strict=True)
would need anyway; any mismatch surfaces as a normal, explicit PyTorch error.

Ablation flags that change only forward()'s CONTROL FLOW without changing any
parameter shape (--disable_exchange, --exchange_detach_source,
--exchange_self_source, --mapper_single_scale, --sketch_self_refine) cannot be
recovered from weights at all. If the checkpoint was trained with any of
these, pass the matching CLI flag explicitly -- otherwise the exported
features will silently use the wrong Photo->Sketch computation even though
the loaded weights are correct. This script prints an explicit warning
whenever any of these flags is active.

This tool only supports checkpoints trained with the DEFAULT architecture
(VisualVisualPromptLearner + SimpleTextPromptLearner x2, i.e.
cfg.no_prompt_learning=False and cfg.use_text_visual_exchange=False). If a
checkpoint is detected as using either alternative architecture, this script
raises NotImplementedError naming the detected mode rather than guessing at
unverified code paths.
"""
import argparse
import re
import types

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.clip import clip as _clip
from src.dataset_retrieval import Sketchy, UNSEEN_CLASSES, ValidDataset
from src.hicropl import SimpleTextPromptLearner, TextEncoder
from src.model_hicropl import CustomCLIP, HiCroPL_SBIR
from src.utils import load_clip_to_cpu

VISUAL_LEARNER_PREFIX = "model.visual_visual_learner."
TEXT_VISUAL_EXCHANGE_PREFIX = "model.text_visual_learner_photo."
NO_PROMPT_LEARNING_KEY = "model.tokenized_prompts_photo"


def build_argparser():
    p = argparse.ArgumentParser(
        description="Read-only diagnostic feature export (no metrics computed)."
    )
    p.add_argument("--ckpt", type=str, default=None,
                    help="Path to a .ckpt saved by ModelCheckpoint. Required unless --frozen.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--frozen", action="store_true",
                    help="Skip --ckpt entirely; use raw pretrained CLIP with a fixed "
                         "template and no prompt learning (mirrors --no_prompt_learning).")
    p.add_argument("--max_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ctx_init", type=str, default="a photo of a")
    p.add_argument("--ctx_init_sketch", type=str, default="a sketch of a")
    # Forward-flow-only ablation flags -- NOT recoverable from checkpoint
    # weights (see module docstring). Ignored entirely when --frozen.
    p.add_argument("--disable_exchange", action="store_true")
    p.add_argument("--exchange_detach_source", action="store_true")
    p.add_argument("--exchange_self_source", action="store_true")
    p.add_argument("--mapper_single_scale", action="store_true")
    p.add_argument("--sketch_self_refine", action="store_true")
    args = p.parse_args()
    if not args.frozen and args.ckpt is None:
        p.error("--ckpt is required unless --frozen is set")
    return args


def infer_visual_visual_arch(state_dict):
    """Infer the VisualVisualPromptLearner architecture (prompt_depth, n_ctx,
    cross_layer, n_proxy, exchange_free_source, sketch_self_refine_ln) purely
    from the checkpoint's own state_dict keys/shapes, since no hyperparameters
    are saved (see module docstring / project audit Q7).

    Raises RuntimeError naming the missing key/prefix if an assumption this
    function relies on does not hold -- never guesses a default silently.
    """
    prefix = VISUAL_LEARNER_PREFIX

    depth_idx = set()
    for k in state_dict:
        m = re.match(re.escape(prefix) + r"cross_prompts_photo\.(\d+)$", k)
        if m:
            depth_idx.add(int(m.group(1)))
    if not depth_idx:
        raise RuntimeError(
            f"No keys matching '{prefix}cross_prompts_photo.<i>' found -- cannot infer "
            f"prompt_depth. This tool assumes the default VisualVisualPromptLearner "
            f"architecture; the checkpoint does not appear to match it."
        )
    prompt_depth = max(depth_idx) + 1

    ctx_photo_key = prefix + "ctx_photo"
    if ctx_photo_key not in state_dict:
        raise RuntimeError(f"Missing key '{ctx_photo_key}' -- cannot infer n_ctx.")
    n_ctx = int(state_dict[ctx_photo_key].shape[0])

    cross_layer_idx = set()
    for k in state_dict:
        m = re.match(re.escape(prefix) + r"attn_pooling_photo_nets\.(\d+)\.", k)
        if m:
            cross_layer_idx.add(int(m.group(1)))
    cross_layer = (max(cross_layer_idx) + 1) if cross_layer_idx else 0

    n_proxy = 1
    proxy_key0 = prefix + "photo_proxy_token.0"
    if cross_layer > 0:
        if proxy_key0 not in state_dict:
            raise RuntimeError(
                f"cross_layer was inferred as {cross_layer} (> 0, from "
                f"attn_pooling_photo_nets.* keys) but '{proxy_key0}' is missing -- "
                f"cannot infer n_proxy."
            )
        n_proxy = int(state_dict[proxy_key0].shape[0])

    exchange_free_source = (prefix + "free_source") in state_dict
    sketch_self_refine_ln = (prefix + "ln_selfrefine.weight") in state_dict

    return dict(
        prompt_depth=prompt_depth,
        n_ctx=n_ctx,
        cross_layer=cross_layer,
        n_proxy=n_proxy,
        exchange_free_source=exchange_free_source,
        sketch_self_refine_ln=sketch_self_refine_ln,
    )


def detect_architecture_mode(state_dict):
    """Return 'default', 'use_text_visual_exchange', or 'no_prompt_learning'
    based on which key prefixes are actually present in the checkpoint.
    Raises RuntimeError if none of the three expected patterns match."""
    has_vv = any(k.startswith(VISUAL_LEARNER_PREFIX) for k in state_dict)
    has_tve = any(k.startswith(TEXT_VISUAL_EXCHANGE_PREFIX) for k in state_dict)
    has_npl = any(k.startswith(NO_PROMPT_LEARNING_KEY) for k in state_dict)

    if has_vv:
        return "default"
    if has_tve:
        return "use_text_visual_exchange"
    if has_npl:
        return "no_prompt_learning"

    top_level = sorted({k.split(".")[1] for k in state_dict if k.startswith("model.")})
    raise RuntimeError(
        f"Checkpoint has none of the expected key prefixes "
        f"('{VISUAL_LEARNER_PREFIX}', '{TEXT_VISUAL_EXCHANGE_PREFIX}', "
        f"'{NO_PROMPT_LEARNING_KEY}') -- cannot determine architecture. "
        f"Top-level 'model.*' attribute names found instead: {top_level}"
    )


def encode_text_frozen(names, template, clip_model):
    """Mirror CustomCLIP.__init__'s no_prompt_learning template construction
    (src/model_hicropl.py:106-110) and forward()'s encode_text call
    (:193-194) exactly -- NO prompt learner involved, matches --frozen
    semantics (raw pretrained CLIP, fixed template)."""
    template_clean = template.replace("_", " ")
    names_clean = [n.replace("_", " ") for n in names]
    prompts = [f"{template_clean} {n}." for n in names_clean]
    device = next(clip_model.parameters()).device
    tokenized = torch.cat([_clip.tokenize(p) for p in prompts]).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(tokenized)
    return feat


def encode_text_for_names(names, clip_model, cfg, trained_learner):
    """Encode `names` through the SAME prompted TextEncoder pipeline as
    training, reusing the ALREADY-LEARNED prompt tensors from
    `trained_learner` (e.g. custom_clip.text_prompt_photo) instead of the
    freshly (re-)initialized ones a throwaway SimpleTextPromptLearner would
    otherwise get. Needed because SimpleTextPromptLearner's token_prefix/
    token_suffix buffers are fixed to a specific classname list at
    construction time (the seen classes used during training), so a
    different classname list (unseen) requires a fresh instance for the
    tokenization/prefix/suffix plumbing -- but the learned context vectors
    themselves must come from the trained checkpoint, not be reinitialized.
    """
    device = next(clip_model.parameters()).device
    learner = SimpleTextPromptLearner(cfg, names, clip_model).to(device)
    learner.cross_prompts_text = trained_learner.cross_prompts_text
    text_input, cross_prompts_text_deeper = learner(label=None)
    text_encoder = TextEncoder(clip_model).to(device)
    with torch.no_grad():
        feat = text_encoder(text_input, learner.tokenized_prompts, cross_prompts_text_deeper)
    return feat


def collect_visual_features(dataset, modality, wrapper, batch_size, workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    feats, labels = [], []
    with torch.no_grad():
        for tensor, label in loader:
            tensor = tensor.to(device)
            feat = wrapper.extract_eval_features(tensor, modality=modality)
            feats.append(feat.cpu())
            labels.append(label.cpu())
    feats = torch.cat(feats, dim=0).numpy().astype(np.float32)
    labels = torch.cat(labels, dim=0).numpy().astype(np.int64)
    return feats, labels


def main():
    args = build_argparser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_ckpt = None
    state_dict = None
    arch = dict(prompt_depth=9, n_ctx=4, cross_layer=-1, n_proxy=1,
                exchange_free_source=False, sketch_self_refine_ln=False)

    if args.frozen:
        print("[diag_export] --frozen set: using raw pretrained CLIP, no prompt "
              "learning, no --ckpt loaded, forward-flow ablation flags ignored.")
    else:
        print(f"[diag_export] Loading checkpoint: {args.ckpt}")
        raw_ckpt = torch.load(args.ckpt, map_location="cpu")
        if "state_dict" not in raw_ckpt:
            raise RuntimeError(
                f"Checkpoint at {args.ckpt} has no 'state_dict' key -- not a "
                f"pytorch-lightning ModelCheckpoint file as expected. "
                f"Top-level keys found: {list(raw_ckpt.keys())}"
            )
        state_dict = raw_ckpt["state_dict"]

        mode = detect_architecture_mode(state_dict)
        if mode != "default":
            raise NotImplementedError(
                f"Checkpoint was detected as trained with architecture mode "
                f"'{mode}' (via key-prefix inspection of state_dict), but this tool "
                f"only supports the default VisualVisualPromptLearner + "
                f"SimpleTextPromptLearner architecture (cfg.no_prompt_learning=False, "
                f"cfg.use_text_visual_exchange=False). Refusing to guess at an "
                f"unverified code path -- extend infer_visual_visual_arch()/main() "
                f"in tools/diag_export.py before using it on this checkpoint. "
                f"(For a no-prompt-learning baseline specifically, use --frozen instead, "
                f"which does not require a checkpoint at all.)"
            )

        arch = infer_visual_visual_arch(state_dict)
        print(f"[diag_export] Inferred architecture from checkpoint state_dict: {arch}")

        if arch["sketch_self_refine_ln"] and not args.disable_exchange:
            print("[diag_export] Detected 'ln_selfrefine' weights (--sketch_self_refine_ln) "
                  "in checkpoint -- forcing --disable_exchange=True (required by "
                  "VisualVisualPromptLearner's own assert for this flag to be valid).")
            args.disable_exchange = True

        forward_flow_flags = dict(
            disable_exchange=args.disable_exchange,
            exchange_detach_source=args.exchange_detach_source,
            exchange_self_source=args.exchange_self_source,
            mapper_single_scale=args.mapper_single_scale,
            sketch_self_refine=args.sketch_self_refine,
        )
        if any(forward_flow_flags.values()):
            print("[diag_export] NOTE: the following ablation flags change only "
                  "forward()'s CONTROL FLOW and are NOT recoverable from checkpoint "
                  "weights (project audit Q7: hyperparameters are never saved). Using "
                  "exactly what was passed on this CLI invocation -- if these do not "
                  "match how the checkpoint was actually trained, the loaded weights "
                  "are correct but the Photo->Sketch computation will silently use the "
                  f"wrong code path: {forward_flow_flags}")

    opts = types.SimpleNamespace(
        backbone="ViT-B/32",
        data_dir=args.data_dir, dataset=args.dataset, max_size=args.max_size, data_split=-1.0,
        prompt_depth=arch["prompt_depth"], n_ctx=arch["n_ctx"], cross_layer=arch["cross_layer"],
        vision_depth=-1, language_depth=-1, vision_ctx=-1, language_ctx=-1,
        clip_trainer="CoOp" if args.frozen else "HiCroPL",
        no_prompt_learning=args.frozen,
        use_text_visual_exchange=False,
        ctx_init=args.ctx_init, ctx_init_sketch=args.ctx_init_sketch,
        n_proxy=arch["n_proxy"], proxy_init="randn",
        exchange_free_source=arch["exchange_free_source"],
        disable_exchange=args.disable_exchange,
        exchange_detach_source=args.exchange_detach_source,
        exchange_self_source=args.exchange_self_source,
        mapper_single_scale=args.mapper_single_scale,
        sketch_self_refine=args.sketch_self_refine,
        sketch_self_refine_ln=arch["sketch_self_refine_ln"],
    )

    if opts.dataset not in UNSEEN_CLASSES:
        print(f"[diag_export] WARNING: --dataset={opts.dataset!r} is not a key in "
              f"UNSEEN_CLASSES ({sorted(UNSEEN_CLASSES.keys())}) -- "
              f"src/dataset_retrieval.py silently falls back to UNSEEN_CLASSES['sketchy'] "
              f"in this case (see dataset_retrieval.py:61,166). Verify this is intended.")

    clip_model = load_clip_to_cpu(opts)
    clip_model.float()
    clip_model.to(device)

    print("[diag_export] Indexing dataset for seen/unseen classnames...")
    train_dataset = Sketchy(opts, Sketchy.data_transform(opts), mode="train")
    seen_names = list(train_dataset.all_categories)

    val_sketch = ValidDataset(opts, mode="sketch")
    val_photo = ValidDataset(opts, mode="photo")
    if val_sketch.all_categories != val_photo.all_categories:
        raise RuntimeError(
            "val_sketch.all_categories != val_photo.all_categories -- sketch/photo "
            "unseen category lists differ (mirrors the same assertion in "
            "experiments/hicropl_prompt.py:79-80)."
        )
    unseen_names = list(val_sketch.all_categories)
    print(f"[diag_export] seen={len(seen_names)} categories, "
          f"unseen={len(unseen_names)} categories.")

    custom_clip = CustomCLIP(opts, clip_model, classnames=seen_names)
    wrapper = HiCroPL_SBIR(cfg=opts, args=opts, classnames=seen_names, model=custom_clip)

    meta = {
        "ckpt": args.ckpt,
        "frozen": bool(args.frozen),
        "epoch": None,
        "inferred_arch": arch,
        "args": vars(args),
        "opts": {k: v for k, v in vars(opts).items()},
        "note": (
            "Architecture (prompt_depth/n_ctx/cross_layer/n_proxy/exchange_free_source/"
            "sketch_self_refine_ln) is INFERRED from checkpoint state_dict shapes, NOT "
            "from saved hyperparameters -- this repo never calls "
            "self.save_hyperparameters() (project audit Q7). Forward-flow-only ablation "
            "flags (disable_exchange/exchange_detach_source/exchange_self_source/"
            "mapper_single_scale/sketch_self_refine) were taken from THIS CLI invocation, "
            "NOT recovered from the checkpoint -- verify they match training. Text "
            "features are exported as *_photo/*_sketch because this architecture always "
            "has two separate text branches (see forward())."
        ),
    }

    if not args.frozen:
        wrapper.load_state_dict(state_dict, strict=True)
        meta["epoch"] = raw_ckpt["epoch"]
        print(f"[diag_export] Loaded checkpoint state_dict OK (epoch={meta['epoch']}).")
    else:
        print("[diag_export] --frozen: using freshly-initialized (pretrained, "
              "non-finetuned) CLIP weights -- no state_dict loaded.")

    wrapper.to(device)
    wrapper.eval()

    # ---- Text features (always two branches: photo / sketch) ----
    with torch.no_grad():
        if args.frozen:
            text_seen_photo = custom_clip.normalize_features(
                custom_clip.clip.encode_text(custom_clip.tokenized_prompts_photo)
            )
            text_seen_sketch = custom_clip.normalize_features(
                custom_clip.clip.encode_text(custom_clip.tokenized_prompts_sketch)
            )
            text_unseen_photo = custom_clip.normalize_features(
                encode_text_frozen(unseen_names, opts.ctx_init, custom_clip.clip)
            )
            text_unseen_sketch = custom_clip.normalize_features(
                encode_text_frozen(unseen_names, opts.ctx_init_sketch, custom_clip.clip)
            )
        else:
            if not hasattr(custom_clip, "text_prompt_photo"):
                raise RuntimeError(
                    "custom_clip has no attribute 'text_prompt_photo' -- expected for "
                    "the default architecture, cannot export text_seen_photo/text_unseen_photo."
                )
            if not hasattr(custom_clip, "text_prompt_sketch"):
                raise RuntimeError(
                    "custom_clip has no attribute 'text_prompt_sketch' -- expected for "
                    "the default architecture, cannot export text_seen_sketch/text_unseen_sketch."
                )

            text_input_photo, deeper_photo = custom_clip.text_prompt_photo(label=None)
            text_seen_photo_raw = custom_clip.text_encoder_photo(
                text_input_photo, custom_clip.text_prompt_photo.tokenized_prompts, deeper_photo
            )
            text_seen_photo = custom_clip.normalize_features(text_seen_photo_raw)

            text_input_sketch, deeper_sketch = custom_clip.text_prompt_sketch(label=None)
            text_seen_sketch_raw = custom_clip.text_encoder_sketch(
                text_input_sketch, custom_clip.text_prompt_sketch.tokenized_prompts, deeper_sketch
            )
            text_seen_sketch = custom_clip.normalize_features(text_seen_sketch_raw)

            cfg_photo = types.SimpleNamespace(prompt_depth=opts.prompt_depth, n_ctx=opts.n_ctx,
                                               ctx_init=opts.ctx_init)
            cfg_sketch = types.SimpleNamespace(prompt_depth=opts.prompt_depth, n_ctx=opts.n_ctx,
                                                ctx_init=opts.ctx_init_sketch)
            text_unseen_photo = custom_clip.normalize_features(
                encode_text_for_names(unseen_names, custom_clip.clip, cfg_photo,
                                       custom_clip.text_prompt_photo)
            )
            text_unseen_sketch = custom_clip.normalize_features(
                encode_text_for_names(unseen_names, custom_clip.clip, cfg_sketch,
                                       custom_clip.text_prompt_sketch)
            )

    text_seen_photo = text_seen_photo.cpu().numpy().astype(np.float32)
    text_seen_sketch = text_seen_sketch.cpu().numpy().astype(np.float32)
    text_unseen_photo = text_unseen_photo.cpu().numpy().astype(np.float32)
    text_unseen_sketch = text_unseen_sketch.cpu().numpy().astype(np.float32)

    # ---- Visual features (unseen split gallery/query) ----
    print("[diag_export] Extracting sketch features (unseen split)...")
    sk_feat, sk_label = collect_visual_features(
        val_sketch, "sketch", wrapper, args.batch_size, args.workers, device
    )
    print("[diag_export] Extracting photo features (unseen split)...")
    ph_feat, ph_label = collect_visual_features(
        val_photo, "photo", wrapper, args.batch_size, args.workers, device
    )

    names_seen = np.array(seen_names, dtype=object)
    names_unseen = np.array(unseen_names, dtype=object)
    meta_arr = np.array(meta, dtype=object)

    arrays = {
        "text_seen_photo": text_seen_photo,
        "text_seen_sketch": text_seen_sketch,
        "text_unseen_photo": text_unseen_photo,
        "text_unseen_sketch": text_unseen_sketch,
        "names_seen": names_seen,
        "names_unseen": names_unseen,
        "sk_feat": sk_feat,
        "sk_label": sk_label,
        "ph_feat": ph_feat,
        "ph_label": ph_label,
        "meta": meta_arr,
    }

    print("[diag_export] Final array shapes/dtypes before saving:")
    for k, v in arrays.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")

    np.savez(args.out, **arrays)
    print(f"[diag_export] Saved to {args.out}")


if __name__ == "__main__":
    main()
