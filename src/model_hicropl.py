import copy
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision

from src.hicropl import (
    TextEncoder,
    VisualEncoder,
    VisualVisualPromptLearner,
    SimpleTextPromptLearner,
    CrossModalPromptLearner,
)


def freeze_model(m):
    """Freeze all parameters of the given module."""
    for param in m.parameters():
        param.requires_grad_(False)


def freeze_all_but_bn(model):
    """Freeze every parameter except those owned by nn.LayerNorm modules.

    Matches by module membership (not attribute name) so it correctly covers
    parameters that aren't literally named `weight`/`bias`, e.g.
    nn.MultiheadAttention's `in_proj_weight`/`in_proj_bias`, or loose
    nn.Parameters like `class_embedding`/`positional_embedding`/`proj`/
    `text_projection`/`logit_scale` — all of which must stay frozen per the
    CLIP-AT design (only LayerNorm trainable; Attention and MLP frozen).
    """
    ln_param_ids = {
        id(p)
        for m in model.modules() if isinstance(m, torch.nn.LayerNorm)
        for p in m.parameters()
    }
    for p in model.parameters():
        if id(p) not in ln_param_ids:
            p.requires_grad_(False)



_MAPPER_NAMES = ('photo2sketch_net', 'sketch2photo_net', 'text2visual_net', 'visual2text_net')


def _classify_group(name):
    """Partition every parameter into exactly one of 4 reporting groups.

    Returns (group, subgroup). Mutually exclusive and jointly exhaustive:
    anything unmatched lands in 'ungrouped', which is printed by name so a new
    module can never be silently absorbed into a total.

        backbone  -- CLIP itself: LayerNorm (intended) + anything else (LEAK)
        tokens    -- learnable prompt tokens, split modality x domain
        exchange  -- LKP / Mapper / proxy tokens / cross-exchange extras
        ungrouped -- everything else (should always be empty)

    Order matters: exchange is tested BEFORE tokens because `photo_proxy_token`
    and `attn_pooling_photo_nets` contain 'photo' and would otherwise be
    misread as photo prompt tokens.
    """
    # Second backbone first -- 'clip_aug.' would also match the 'clip' checks
    # below if they were reordered.
    #
    # Backbone subgroups are just which backbone it is. What KIND of param it is
    # is not encoded here: the design says a trainable backbone param must be a
    # LayerNorm, so that is an invariant to assert, not a category to tabulate.
    # log_param_breakdown checks it and warns by name.
    if name.startswith('clip_aug.'):
        return ('backbone', 'aug')
    if name.startswith('clip.') or '_encoder_photo.' in name or '_encoder_sketch.' in name \
            or name == 'logit_scale':
        return ('backbone', 'main')

    if 'attn_pooling' in name:
        return ('exchange', 'lkp')
    if any(k in name for k in _MAPPER_NAMES):
        return ('exchange', 'mapper')
    if 'proxy_token' in name:
        return ('exchange', 'proxy_token')
    if 'free_source' in name or 'ln_selfrefine' in name:
        return ('exchange', 'other')

    if 'cross_prompts_text' in name or name.endswith('.ctx'):
        modality = 'text'
    elif ('cross_prompts_visual' in name or 'cross_prompts_photo' in name
          or 'cross_prompts_sketch' in name or '.ctx_photo' in name or '.ctx_sketch' in name):
        modality = 'visual'
    else:
        return ('ungrouped', '?')
    owner = name.split('.')[0]
    if 'photo' in owner or '_photo' in name or 'ctx_photo' in name:
        domain = 'photo'
    elif 'sketch' in owner or '_sketch' in name or 'ctx_sketch' in name:
        domain = 'sketch'
    else:
        domain = '?'
    return ('tokens', f'{modality}/{domain}')


# Whether CustomCLIP.forward runs the clip_aug encoder with autograd enabled.
# Flip this to False if a torch.no_grad() is ever put back around that call --
# on_after_backward compares this prediction against real gradients and warns
# if they disagree, so a stale value cannot go unnoticed for long.
_AUG_FORWARD_BUILDS_GRAPH = True

# Trainable params clip_aug is supposed to expose: the LayerNorms of the
# ViT-B/32 VISUAL tower only (__init__ scopes unfreeze_ln to .visual).
#   12 blocks x (ln_1 + ln_2) x (weight + bias) x 768 = 36,864
#   ln_pre + ln_post                x (weight + bias) x 768 =  3,072
# Verified against a real build. Any other number means the freeze scope moved.
_AUG_LN_EXPECTED = 39_936


def _is_idle(name, cfg, group):
    """Is this parameter guaranteed to receive NO gradient this run?

    requires_grad=True only means "the optimizer holds it" -- flags and forward
    branches cut gradient paths without removing modules, so a flag like
    --disable_exchange leaves ~37M params in the optimizer whose .grad stays
    None forever. Counting those as trainable capacity is the exact misreading
    this column exists to prevent.

    Covers both the exchange group (derived from the forward() branches in
    src/hicropl.py:614-676) and the backbone group. Returns a short reason
    string, or None if the parameter is live.
    """
    if group == 'backbone':
        # main: LayerNorm sits on the path of every loss term -- always live.
        # aug: live only because the clip_aug call is NOT wrapped in no_grad.
        #      Under no_grad these 65,536 params would be pure dead weight in
        #      the optimizer while still reporting requires_grad=True.
        if name.startswith('clip_aug.'):
            if not _AUG_FORWARD_BUILDS_GRAPH:
                return 'clip_aug forward under no_grad'
            # Canary. The aug branch calls encode_image only, so a trainable
            # text-tower LayerNorm here could never receive gradient. __init__
            # scopes the unfreeze to clip_aug.visual precisely so this cannot
            # happen -- reaching this line means someone widened it back to
            # freeze_all_but_bn(self.clip_aug).
            if not name.startswith('clip_aug.visual.'):
                return 'clip_aug text tower never called (encode_image only)'
        return None

    if group != 'exchange':
        return None

    if getattr(cfg, 'use_text_visual_exchange', False):
        return None  # --disable_exchange has no effect on this architecture

    if getattr(cfg, 'disable_exchange', False):
        # Both mapping blocks are skipped entirely. The only survivors are the
        # modules the self-refine branches still call (src/hicropl.py:657-672).
        if 'photo2sketch_net' in name and (getattr(cfg, 'sketch_self_refine', False)
                                           or getattr(cfg, 'sketch_self_refine_ln', False)):
            return None  # still live via the query side (src/hicropl.py:672)
        if 'ln_selfrefine' in name:
            # Run D applies it ONLY on the k/v side, which is detached at
            # src/hicropl.py:671 -- so this LayerNorm never trains and stays at
            # its init (weight=1, bias=0) for the whole run.
            return 'sketch_self_refine_ln (k/v detached)'
        return 'disable_exchange'

    # Photo->Sketch block: the photo-side LKP output is detached (or skipped),
    # so attn_pooling_photo_nets + photo_proxy_token never get gradient.
    photo_lkp = 'attn_pooling_photo_nets' in name or 'photo_proxy_token' in name
    if photo_lkp:
        if getattr(cfg, 'exchange_free_source', False):
            return 'exchange_free_source'   # module not even called
        if getattr(cfg, 'exchange_detach_source', False):
            return 'exchange_detach_source'
        if getattr(cfg, 'exchange_self_source', False):
            return 'exchange_self_source'
    return None


def _fmt_table(headers, rows, printer):
    """Minimal ASCII table -- no external deps, right-aligns numeric columns."""
    if not rows:
        printer("    (empty)")
        return
    cols = list(zip(*([headers] + [[str(c) for c in r] for r in rows])))
    widths = [max(len(c) for c in col) for col in cols]
    numeric = [all(c.replace(',', '').replace('-', '').isdigit() or c == ''
                   for c in col[1:]) for col in cols]

    def line(cells):
        return "    " + "  ".join(
            c.rjust(widths[i]) if numeric[i] else c.ljust(widths[i])
            for i, c in enumerate(cells)
        )

    printer(line(headers))
    printer("    " + "  ".join('-' * w for w in widths))
    for r in rows:
        printer(line([str(c) for c in r]))


def log_param_breakdown(model, printer=print):
    """Print trainable params grouped as backbone / tokens / exchange.

    Every total is deduped by id(p). named_parameters(remove_duplicate=False) is
    used deliberately: the single shared backbone makes each CLIP tensor
    reachable under 3 names, so double counting is a live hazard here.

    The `state` column separates DECLARED capacity (requires_grad=True, in the
    optimizer) from EFFECTIVE capacity (actually reachable by gradient). Under
    --disable_exchange the two differ by ~37M params.
    """
    cfg = getattr(model, 'cfg', None)
    ln_ids = {id(p) for m in model.modules() if isinstance(m, nn.LayerNorm) for p in m.parameters()}

    seen, recs = set(), []
    for name, p in model.named_parameters(remove_duplicate=False):
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        group, sub = _classify_group(name)
        idle = _is_idle(name, cfg, group)
        recs.append({'name': name, 'numel': p.numel(), 'group': group,
                     'sub': sub, 'idle': idle, 'is_ln': id(p) in ln_ids})

    printer("")
    printer("=" * 78)
    printer("[2] TRAINABLE BY GROUP (backbone / tokens / exchange)")
    printer("=" * 78)

    agg = {}
    for r in recs:
        key = (r['group'], r['sub'], r['idle'] or '')
        n, s = agg.get(key, (0, 0))
        agg[key] = (n + 1, s + r['numel'])

    rows, totals, live_totals = [], {}, {}
    for g in ['backbone', 'tokens', 'exchange', 'ungrouped']:
        subs = sorted([kv for kv in agg.items() if kv[0][0] == g], key=lambda kv: -kv[1][1])
        totals[g] = sum(v[1] for _, v in subs)
        live_totals[g] = sum(v[1] for k, v in subs if not k[2])
        if not subs:
            continue
        for (_, sub, idle), (n, s) in subs:
            rows.append([g, sub, 'IDLE' if idle else 'active', n, f"{s:,}"])
        rows.append([f"-> {g} TOTAL", "", "", sum(v[0] for _, v in subs), f"{totals[g]:,}"])
    grand = sum(totals.values())
    live = sum(live_totals.values())
    rows.append(["== GRAND TOTAL", "", "", len(recs), f"{grand:,}"])
    _fmt_table(["group", "subgroup", "state", "n_tensors", "numel"], rows, printer)

    printer(f"    DECLARED  (requires_grad, in optimizer): {grand:,}")
    if live != grand:
        idle_by_reason = {}
        for r in recs:
            if r['idle']:
                n, s = idle_by_reason.get(r['idle'], (0, 0))
                idle_by_reason[r['idle']] = (n + 1, s + r['numel'])
        printer(f"    EFFECTIVE (gradient actually reaches):  {live:,}")
        for reason, (n, s) in sorted(idle_by_reason.items(), key=lambda kv: -kv[1][1]):
            printer(f"    IDLE -- --{reason}: {n} tensors, {s:,} params "
                    f"({100.0 * s / grand:.1f}% of declared) never receive gradient")
    else:
        printer(f"    EFFECTIVE (gradient actually reaches):  {live:,}  -- no idle params")

    printer(f"    check: backbone {totals['backbone']:,} + tokens {totals['tokens']:,} + "
            f"exchange {totals['exchange']:,} + ungrouped {totals['ungrouped']:,} = {grand:,} -> "
            f"{'OK' if grand == sum(r['numel'] for r in recs) else 'MISMATCH'}")
    if totals['ungrouped']:
        printer(f"    WARNING -- {totals['ungrouped']:,} params khong thuoc nhom nao:")
        for r in [x for x in recs if x['group'] == 'ungrouped'][:20]:
            printer(f"        {r['name']}  {r['numel']:,}")
    # Invariant: a trainable backbone param must belong to a LayerNorm. The main
    # backbone is LN-only by CLIP-AT design (freeze_all_but_bn); the aug backbone
    # is frozen outright, so it should have no trainable param at all. Anything
    # else is a leak -- name it rather than let it sit inside a subtotal.
    leaks = [r for r in recs if r['group'] == 'backbone' and not r['is_ln']]
    if leaks:
        printer(f"    WARNING -- BACKBONE LEAK: {sum(r['numel'] for r in leaks):,} params "
                f"ngoai LayerNorm dang trainable ({len(leaks)} tensors):")
        for r in leaks[:20]:
            printer(f"        [{r['sub']}] {r['name']}  {r['numel']:,}")
        if len(leaks) > 20:
            printer(f"        ... {len(leaks) - 20} more")
    # The table only lists trainable params, so the aug backbone's ~151M frozen
    # weights would otherwise be invisible in the only param log there is.
    clip_aug = getattr(model, 'clip_aug', None)
    shared_enc = bool(getattr(cfg, 'aug_shared_encoder', False))
    ident_tf = bool(getattr(cfg, 'aug_identity_transform', False))
    detach_view = bool(getattr(cfg, 'aug_detach_view', False))
    if clip_aug is None:
        reason = ('--aug_shared_encoder (Run A: aug views ride the main encoder)'
                  if shared_enc else '--disable_aug_branch')
        printer(f"    aug backbone: absent ({reason})")
    else:
        tot = sum(p.numel() for p in clip_aug.parameters())
        tr = sum(p.numel() for p in clip_aug.parameters() if p.requires_grad)
        note = ' -- visual-tower LayerNorm trainable' if tr else ' -- fully frozen'
        printer(f"    aug backbone: {tot:,} params, {tr:,} trainable{note}")
        if tr and not _AUG_FORWARD_BUILDS_GRAPH:
            printer("        WARNING: forward runs clip_aug under no_grad, so these "
                    "never update -- freeze_all_but_bn has no effect")
    printer(f"GROUP_FP | backbone={totals['backbone']} | tokens={totals['tokens']} | "
            f"exchange={totals['exchange']} | ungrouped={totals['ungrouped']} | "
            f"declared={grand} | effective={live}")

    # ---- Run A / Run B fingerprint + hard reference checks -------------------
    # Subtotals below are DISJOINT and sum back to `declared`, so a silent
    # regrouping cannot hide params inside a total that still looks right.
    def _sum(pred):
        return sum(r['numel'] for r in recs if pred(r))

    n_clip_aug = _sum(lambda r: r['group'] == 'backbone' and r['sub'] == 'aug')
    n_ln = _sum(lambda r: r['group'] == 'backbone' and r['sub'] == 'main')
    n_prompt = totals['tokens']
    n_mapper = _sum(lambda r: r['group'] == 'exchange' and r['sub'] == 'mapper')
    n_lkp = _sum(lambda r: r['group'] == 'exchange' and r['sub'] == 'lkp')
    n_other = grand - (n_clip_aug + n_ln + n_prompt + n_mapper + n_lkp)

    printer(f"RUN_MODE  | aug_shared_encoder={int(shared_enc)} | "
            f"aug_identity_transform={int(ident_tf)} | "
            f"aug_detach_view={int(detach_view)} | "
            f"clip_aug_loaded={'yes' if clip_aug is not None else 'no'}")
    printer(f"PARAM_FP  | trainable={grand} | ln={n_ln} | prompt={n_prompt} | "
            f"mapper={n_mapper} | lkp={n_lkp} | clip_aug={n_clip_aug} | other={n_other}")

    # Run A reference: the second backbone must not exist at all -- not built,
    # not in the optimizer, not in the checkpoint.
    if shared_enc:
        if n_clip_aug == 0 and clip_aug is None:
            printer("    CHECK Run A: clip_aug=0, clip_aug_loaded=no -- OK")
        else:
            printer(f"    ERROR Run A: expected clip_aug=0 and clip_aug_loaded=no, got "
                    f"clip_aug={n_clip_aug} and clip_aug_loaded="
                    f"{'yes' if clip_aug is not None else 'no'}")
    # Run B (and the plain aug branch) reference: exactly the visual tower's
    # LayerNorms -- 12 blocks x 2 LN x (weight+bias) x 768 = 36,864, plus
    # ln_pre + ln_post = 3,072.
    if clip_aug is not None:
        if n_clip_aug == _AUG_LN_EXPECTED:
            printer(f"    CHECK Run B: clip_aug trainable={n_clip_aug:,} -- OK "
                    f"(visual-tower LayerNorm only)")
        else:
            printer(f"    WARNING Run B: clip_aug trainable={n_clip_aug:,}, expected "
                    f"{_AUG_LN_EXPECTED:,} (ViT-B/32 visual tower: 12x2x2x768=36,864 "
                    f"+ ln_pre/ln_post=3,072). Breakdown by tensor:")
            for r in [x for x in recs if x['group'] == 'backbone' and x['sub'] == 'aug'][:60]:
                printer(f"        {r['name']}  {r['numel']:,}  "
                        f"{'LN' if r['is_ln'] else 'NOT-LN'}")
    printer("=" * 78)
    printer("")



def unfreeze_ln(m):
    """Mở lại weight/bias của mọi LayerNorm trong module.

    Dùng SAU `freeze_model(...)` để thực thi pattern "chỉ LN trainable":
        freeze_model(encoder)           # đông cứng tất cả
        encoder.apply(unfreeze_ln)      # chỉ mở LN
    """
    if isinstance(m, nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(True)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(True)


class CustomCLIP(nn.Module):
    """
    HiCroPL-SBIR Architecture Wrapper.
    Sử dụng HiCroPLFeatureExtractor làm nòng cốt.
    """

    def __init__(self, cfg, clip_model, classnames=None, sample_photo_images=None, sample_sketch_images=None):
        super().__init__()
        self.cfg = cfg
        # Ablation: no visual/text prompt learning at all -- only LayerNorm
        # trainable (matches ducta/baseline's CLIP-AT recipe). Requires
        # clip_model to already be a vanilla (non-prompted) build --
        # experiments/hicropl_prompt.py forces clip_trainer='CoOp' before
        # load_clip_to_cpu when this flag is set, so `self.clip.encode_image`/
        # `encode_text` work standalone (no prompt tensors required).
        self.no_prompt_learning = getattr(cfg, 'no_prompt_learning', False)
        # Alternative architecture: per-branch text<->visual exchange (see
        # CrossModalPromptLearner) instead of the default photo<->sketch
        # VisualVisualPromptLearner. Mutually exclusive with the default path
        # -- --disable_exchange has NO effect here (it only gates the
        # photo<->sketch mapping blocks inside VisualVisualPromptLearner,
        # which isn't constructed at all when this is set).
        self.use_text_visual_exchange = getattr(cfg, 'use_text_visual_exchange', False)

        if classnames is None:
            classnames = []
        if len(classnames) == 0:
            raise ValueError("CustomCLIP requires non-empty classnames during initialization.")

        original_device = next(clip_model.parameters()).device
        self.dtype = clip_model.dtype

        # 1. Single shared backbone for both photo and sketch (matches ducta/baseline:
        # one CLIP copy, same LayerNorm weights updated by gradients from both modalities).
        self.clip = copy.deepcopy(clip_model).to(original_device)
        freeze_all_but_bn(self.clip)
        # Param counts are reported once by log_param_breakdown() in
        # configure_optimizers -- the single source of truth.

        # Single shared logit scale (matches ducta/baseline)
        self.logit_scale = self.clip.logit_scale

        if self.no_prompt_learning:
            print("[ABLATION] no_prompt_learning=True: skipping ALL prompt learners. "
                  "Only LayerNorm is trainable; text uses the fixed ctx_init/ctx_init_sketch template.")
            from src.clip import clip as _clip
            classnames_clean = [name.replace("_", " ") for name in classnames]
            ctx_init_photo = getattr(cfg, 'ctx_init', 'a photo of a').replace("_", " ")
            ctx_init_sketch = getattr(cfg, 'ctx_init_sketch', 'a sketch of a').replace("_", " ")
            prompts_photo = [f"{ctx_init_photo} {name}." for name in classnames_clean]
            prompts_sketch = [f"{ctx_init_sketch} {name}." for name in classnames_clean]
            # Fixed (non-learnable) tokenized templates -- registered as buffers,
            # not nn.Parameter, so they never appear in configure_optimizers.
            self.register_buffer(
                "tokenized_prompts_photo",
                torch.cat([_clip.tokenize(p) for p in prompts_photo]).to(original_device),
            )
            self.register_buffer(
                "tokenized_prompts_sketch",
                torch.cat([_clip.tokenize(p) for p in prompts_sketch]).to(original_device),
            )
        elif self.use_text_visual_exchange:
            # Per-branch bidirectional text<->visual exchange -- ONLY text and
            # visual of the SAME domain ever interact. Two fully independent
            # instances (separate weights, no shared modules, no coupling):
            # text_visual_learner_photo only ever sees photo text + photo
            # visual; text_visual_learner_sketch only ever sees sketch text +
            # sketch visual. Neither instance references the other, and
            # visual_visual_learner/text_prompt_photo/text_prompt_sketch are
            # NOT constructed in this branch at all.
            print("Initializing Photo Text<->Visual Exchange Learner...")
            cfg_photo = copy.copy(cfg)
            cfg_photo.ctx_init = getattr(cfg, 'ctx_init', 'a photo of a')
            self.text_visual_learner_photo = CrossModalPromptLearner(
                cfg_photo, classnames, self.clip, sample_images=sample_photo_images
            )

            print("Initializing Sketch Text<->Visual Exchange Learner...")
            cfg_sketch = copy.copy(cfg)
            cfg_sketch.ctx_init = getattr(cfg, 'ctx_init_sketch', 'a sketch of a')
            self.text_visual_learner_sketch = CrossModalPromptLearner(
                cfg_sketch, classnames, self.clip, sample_images=sample_sketch_images
            )

            self.text_encoder_photo = TextEncoder(self.clip)
            self.text_encoder_sketch = TextEncoder(self.clip)
            self.visual_encoder_photo = VisualEncoder(self.clip)
            self.visual_encoder_sketch = VisualEncoder(self.clip)
        else:
            # -- Prompt Learners --
            # Initialize Visual-Visual learner + simple text learners + adapters
            print("Initializing Visual Prompt Learner (photo + sketch, independent)...")
            self.visual_visual_learner = VisualVisualPromptLearner(
                cfg, self.clip, self.clip,
                sample_photo_images=sample_photo_images, sample_sketch_images=sample_sketch_images
            )

            print("Initializing Photo Text Prompt Learner...")
            cfg_photo = copy.copy(cfg)
            cfg_photo.ctx_init = getattr(cfg, 'ctx_init', 'a photo of a')
            self.text_prompt_photo = SimpleTextPromptLearner(cfg_photo, classnames, self.clip)

            print("Initializing Sketch Text Prompt Learner...")
            cfg_sketch = copy.copy(cfg)
            cfg_sketch.ctx_init = getattr(cfg, 'ctx_init_sketch', 'a sketch of a')
            self.text_prompt_sketch = SimpleTextPromptLearner(cfg_sketch, classnames, self.clip)

            # -- Encoders (both branches wrap the SAME shared backbone) --
            self.text_encoder_photo = TextEncoder(self.clip)
            self.text_encoder_sketch = TextEncoder(self.clip)
            self.visual_encoder_photo = VisualEncoder(self.clip)
            self.visual_encoder_sketch = VisualEncoder(self.clip)

        # -- Augmentation branch: second backbone, built LAST on purpose --
        #
        # ORDER IS LOad-BEARING. load_clip_to_cpu constructs a CLIP with random
        # init before loading pretrained weights, so it consumes the global RNG.
        # Building it earlier would shift the stream for every prompt learner
        # above and silently reroll their init -- exactly the confound that
        # --disable_exchange hit before (see VisualVisualPromptLearner's
        # build-then-discard comment). Built last, enabling or disabling this
        # branch leaves every other parameter bit-identical under a given seed.
        #
        # Vanilla build (clip_trainer='CoOp'): self.clip is a
        # VisionTransformer_HiCroPL whose forward() REQUIRES prompt arguments,
        # so encode_image() raises on it. A prompt-free reference genuinely
        # needs its own non-prompted build; sharing self.clip is not an option.
        self.disable_aug_branch = getattr(cfg, 'disable_aug_branch', False)
        # Run A (--aug_shared_encoder): the augmented views go through the MAIN
        # encoder, so the second backbone is not merely idle -- it must not
        # exist. Skipping load_clip_to_cpu here keeps its 151M params out of the
        # optimizer, out of the checkpoint and out of VRAM. Safe to skip
        # precisely because this block is built LAST (see above): the RNG it
        # would have consumed comes after every other module's init, so every
        # other parameter stays bit-identical to a run that builds it.
        self.aug_shared_encoder = getattr(cfg, 'aug_shared_encoder', False)
        self.aug_detach_view = getattr(cfg, 'aug_detach_view', False)
        # Run B (--aug_identity_transform): clip_aug is built exactly as usual;
        # only the dataset-side transform changes (src/dataset_retrieval.py).
        # One-shot bit-equality check, armed here and fired on the first batch.
        self.aug_identity_transform = getattr(cfg, 'aug_identity_transform', False)
        self._identity_transform_checked = False
        # Explicit None (rather than a missing attribute) so every consumer can
        # use a plain `is not None` test.
        self.clip_aug = None
        if not self.disable_aug_branch and not self.aug_shared_encoder:
            from src.utils import load_clip_to_cpu
            cfg_aug = copy.copy(cfg)
            cfg_aug.clip_trainer = 'CoOp'
            self.clip_aug = load_clip_to_cpu(cfg_aug).to(original_device)
            # Freeze everything, then reopen LayerNorm in the VISUAL TOWER ONLY.
            #
            # freeze_all_but_bn(self.clip_aug) would be the obvious call, but it
            # opens LayerNorm in both towers -- and this branch only ever calls
            # encode_image(). The 50 text-tower LayerNorms (25,600 params) would
            # sit in the optimizer with .grad = None for the entire run: real
            # dead weight, and enough to put an IDLE row in the param log of
            # every experiment, including the exchange ablations that are
            # otherwise clean. Scoping the unfreeze to .visual keeps declared ==
            # effective everywhere.
            freeze_model(self.clip_aug)
            self.clip_aug.visual.apply(unfreeze_ln)
            self.clip_aug.eval()

    def train(self, mode=True):
        """Keep clip_aug in eval mode permanently.

        nn.Module.train() recurses into children, so without this override
        Lightning would flip the frozen reference into train mode at every
        epoch start. It has no dropout/BN, but eval() also documents intent.
        """
        super().train(mode)
        if getattr(self, 'clip_aug', None) is not None:
            self.clip_aug.eval()
        return self

    def normalize_features(self, feat_prenorm):
        """L2-normalize feature tensors."""
        return feat_prenorm / feat_prenorm.norm(dim=-1, keepdim=True)

    def forward(self, x, classnames):
        """
        Forward pass for training with optimized redundancy.
        Calls visual learner ONCE and routes prompts by branch.
        """
        # 7 entries = augmentation branch on; the two augmented views are
        # appended last so this stays backward compatible with the 4/5 forms.
        sk_aug_tensor = photo_aug_tensor = None
        if len(x) == 7:
            sk_tensor, photo_tensor, neg_tensor, label, _filename, sk_aug_tensor, photo_aug_tensor = x
        elif len(x) == 5:
            sk_tensor, photo_tensor, neg_tensor, label, _filename = x
        else:
            sk_tensor, photo_tensor, neg_tensor, label = x[:4]

        # Run B canary: fires once, on the first batch that carries aug tensors.
        if (self.aug_identity_transform and not self._identity_transform_checked
                and photo_aug_tensor is not None):
            self._identity_transform_checked = True
            same_photo = torch.allclose(photo_aug_tensor, photo_tensor)
            same_sketch = torch.allclose(sk_aug_tensor, sk_tensor)
            if same_photo and same_sketch:
                print("IDENTITY TRANSFORM: OK")
            else:
                d_p = (photo_aug_tensor - photo_tensor).abs().max().item()
                d_s = (sk_aug_tensor - sk_tensor).abs().max().item()
                print(f"IDENTITY TRANSFORM: FAIL -- max abs diff photo={d_p:.6e} "
                      f"sketch={d_s:.6e} (expected 0.0 for both)")

        # Run A: the augmented views ride the main encoder. Computed inside each
        # architecture branch below, because the prompt tensors they must reuse
        # are branch-local -- recomputing them here would be a different (freshly
        # called) learner output, and calling encode_image() instead would skip
        # the prompts entirely.
        image_features_photo_aug = image_features_sketch_aug = None
        run_shared_aug = self.aug_shared_encoder and photo_aug_tensor is not None

        if self.no_prompt_learning:
            # Plain frozen CLIP forward (only LayerNorm trainable) -- no
            # prompt tensors of any kind, text uses the fixed template.
            image_features_photo = self.clip.encode_image(photo_tensor.type(self.dtype))
            image_features_sketch = self.clip.encode_image(sk_tensor.type(self.dtype))
            image_features_neg = self.clip.encode_image(neg_tensor.type(self.dtype))
            text_features_all_photo = self.clip.encode_text(self.tokenized_prompts_photo)
            text_features_all_sketch = self.clip.encode_text(self.tokenized_prompts_sketch)
            if run_shared_aug:
                image_features_photo_aug = self.clip.encode_image(photo_aug_tensor.type(self.dtype))
                image_features_sketch_aug = self.clip.encode_image(sk_aug_tensor.type(self.dtype))
        elif self.use_text_visual_exchange:
            # Each branch's learner performs its OWN bidirectional text<->visual
            # exchange -- no coupling between the two learners/branches.
            text_input_photo_all, vis_shallow_photo, cross_prompts_text_deeper_photo, vis_deeper_photo = self.text_visual_learner_photo()
            text_features_all_photo = self.text_encoder_photo(text_input_photo_all, self.text_visual_learner_photo.tokenized_prompts, cross_prompts_text_deeper_photo)
            image_features_photo = self.visual_encoder_photo(photo_tensor.type(self.dtype), vis_shallow_photo, vis_deeper_photo)

            text_input_sketch_all, vis_shallow_sketch, cross_prompts_text_deeper_sketch, vis_deeper_sketch = self.text_visual_learner_sketch()
            text_features_all_sketch = self.text_encoder_sketch(text_input_sketch_all, self.text_visual_learner_sketch.tokenized_prompts, cross_prompts_text_deeper_sketch)
            image_features_sketch = self.visual_encoder_sketch(sk_tensor.type(self.dtype), vis_shallow_sketch, vis_deeper_sketch)

            # Negative branch (uses photo encoder + photo visual prompts)
            image_features_neg = self.visual_encoder_photo(neg_tensor.type(self.dtype), vis_shallow_photo, vis_deeper_photo)

            if run_shared_aug:
                # Same encoders, same prompt tensors as the clean views above.
                image_features_photo_aug = self.visual_encoder_photo(photo_aug_tensor.type(self.dtype), vis_shallow_photo, vis_deeper_photo)
                image_features_sketch_aug = self.visual_encoder_sketch(sk_aug_tensor.type(self.dtype), vis_shallow_sketch, vis_deeper_sketch)
        else:
            # 1. Call visual-visual learner ONCE (shared by both branches)
            photo_shallow, sketch_shallow, photo_deeper, sketch_deeper = self.visual_visual_learner()

            # 2. Photo branch: text learner + visual routing
            # Compute text features for ALL classes (not just batch) - needed for loss computation
            text_input_photo_all, cross_prompts_text_deeper_photo = self.text_prompt_photo(label=None)  # All classes
            text_features_all_photo = self.text_encoder_photo(text_input_photo_all, self.text_prompt_photo.tokenized_prompts, cross_prompts_text_deeper_photo)
            image_features_photo = self.visual_encoder_photo(photo_tensor.type(self.dtype), photo_shallow, photo_deeper)

            # 3. Sketch branch: text learner + visual routing
            # Compute text features for ALL classes (not just batch) - needed for loss computation
            text_input_sketch_all, cross_prompts_text_deeper_sketch = self.text_prompt_sketch(label=None)  # All classes
            text_features_all_sketch = self.text_encoder_sketch(text_input_sketch_all, self.text_prompt_sketch.tokenized_prompts, cross_prompts_text_deeper_sketch)
            image_features_sketch = self.visual_encoder_sketch(sk_tensor.type(self.dtype), sketch_shallow, sketch_deeper)

            # 4. Negative branch (uses photo encoder + photo visual prompts)
            image_features_neg = self.visual_encoder_photo(neg_tensor.type(self.dtype), photo_shallow, photo_deeper)

            # 4b. Run A: augmented views, SAME encoders and SAME prompt tensors
            # (photo_shallow/photo_deeper, sketch_shallow/sketch_deeper) as
            # steps 2-3 -- the learner is not called a second time, so the only
            # thing that differs from the clean pass is the input tensor.
            if run_shared_aug:
                image_features_photo_aug = self.visual_encoder_photo(photo_aug_tensor.type(self.dtype), photo_shallow, photo_deeper)
                image_features_sketch_aug = self.visual_encoder_sketch(sk_aug_tensor.type(self.dtype), sketch_shallow, sketch_deeper)

        # 5. Normalize features
        photo_feat = image_features_photo / image_features_photo.norm(dim=-1, keepdim=True)
        sketch_feat = image_features_sketch / image_features_sketch.norm(dim=-1, keepdim=True)
        neg_feat = image_features_neg / image_features_neg.norm(dim=-1, keepdim=True)
        text_feat_photo = text_features_all_photo / text_features_all_photo.norm(dim=-1, keepdim=True)
        text_feat_sketch = text_features_all_sketch / text_features_all_sketch.norm(dim=-1, keepdim=True)

        # 6. Compute logits
        logit_scale = self.logit_scale.exp()
        logits_photo = logit_scale * photo_feat @ text_feat_photo.t()
        logits_sketch = logit_scale * sketch_feat @ text_feat_sketch.t()

        # 7. Augmentation branch: vanilla CLIP over the augmented views.
        #
        # NO torch.no_grad() HERE -- deliberately. clip_aug is frozen by
        # freeze_all_but_bn, so its LayerNorm (65,536 params) is trainable and
        # the InfoNCE terms are meant to train it. Wrapping this in no_grad
        # would build no autograd graph, those params would get .grad = None,
        # and Adam would skip them: the branch would look trainable in the log
        # while being numerically identical to a fully frozen one. That is
        # exactly what commits fbe44ad and 58c05d6 produced -- identical
        # results despite the flag change.
        #
        # Cost of keeping the graph: activations for two full ViT forwards are
        # retained until backward. _is_idle() encodes the no_grad-free
        # assumption for the log, and on_after_backward re-checks it against
        # real gradients once, so the two can never drift apart silently.
        #
        # Run A (--aug_shared_encoder) replaces this block: the features were
        # already produced above by the main encoder, so all that is left is the
        # same L2 normalization. loss_fn_hicropl sees the identical tuple shape
        # either way, so loss_aug keeps its exact structure and 1.0 coefficient.
        photo_aug_feat = sketch_aug_feat = None
        if image_features_photo_aug is not None:
            photo_aug_feat = image_features_photo_aug / image_features_photo_aug.norm(dim=-1, keepdim=True)
            sketch_aug_feat = image_features_sketch_aug / image_features_sketch_aug.norm(dim=-1, keepdim=True)
            if self.aug_detach_view:
                # One-way variant: the aug view becomes a fixed target, only the
                # clean view is pulled. OFF by default -- the clip_aug branch it
                # is being compared against is symmetric (its visual LayerNorms
                # do receive gradient from loss_aug), so a symmetric Run A is the
                # apples-to-apples setting.
                photo_aug_feat = photo_aug_feat.detach()
                sketch_aug_feat = sketch_aug_feat.detach()
        elif photo_aug_tensor is not None and self.clip_aug is not None:
            f_p = self.clip_aug.encode_image(photo_aug_tensor.type(self.dtype))
            f_s = self.clip_aug.encode_image(sk_aug_tensor.type(self.dtype))
            photo_aug_feat = f_p / f_p.norm(dim=-1, keepdim=True)
            sketch_aug_feat = f_s / f_s.norm(dim=-1, keepdim=True)

        return (
            photo_feat, logits_photo,
            sketch_feat, logits_sketch,
            neg_feat, label,
            text_feat_photo, text_feat_sketch,
            photo_aug_feat, sketch_aug_feat,
        )


class HiCroPL_SBIR(pl.LightningModule):
    def __init__(self, cfg, args, classnames, model):
        super().__init__()
        self.cfg = cfg
        self.args = args
        self.classnames = classnames
        self.model = model
        
        self.best_metric = 1e-3
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)

        self.test_photo_features = []
        self.test_sketch_features = []
        self.test_photo_labels = []
        self.test_sketch_labels = []

    def on_train_epoch_start(self):
        # NOTE: Encoders stay in training mode (required for LayerNorm to use batch statistics)
        # Setting eval() here would conflict with forward() expectation and break BN/LN behavior
        #
        # Own accumulator for the loss breakdown. Reading it back out of
        # trainer.callback_metrics is NOT equivalent: an un-suffixed key there
        # holds whatever was written last, which for a metric logged
        # on_step=True is the LAST STEP's value, while an on_epoch-only metric
        # is written just once per epoch -- so the two are neither the same
        # reduction nor the same epoch. Accumulating here keeps the three terms
        # and their total on one clock.
        self._loss_parts_sum = {}
        self._loss_parts_n = 0

    def on_fit_start(self):
        """Intentionally empty.

        Used to print per-branch learnable-token counts; that information is now
        covered (in params, not token counts) by log_param_breakdown() in
        configure_optimizers, which is the single source of truth.
        """
        pass

    def configure_optimizers(self):
        def add_unique_params(candidates, out_list, seen_ids):
            for p in candidates:
                if p.requires_grad and id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    out_list.append(p)

        seen_ids = set()

        prompt_params = []
        # Collect from whichever learner set is active (mutually exclusive).
        # These include all their internal params (CrossPromptAttention, AttentionPooling, etc.)
        # No learner submodules exist when no_prompt_learning=True (only
        # LayerNorm is trainable in that mode).
        if self.model.no_prompt_learning:
            learner_modules = set()
        elif self.model.use_text_visual_exchange:
            add_unique_params(self.model.text_visual_learner_photo.parameters(), prompt_params, seen_ids)
            add_unique_params(self.model.text_visual_learner_sketch.parameters(), prompt_params, seen_ids)
            learner_modules = {'text_visual_learner_photo', 'text_visual_learner_sketch'}
        else:
            add_unique_params(self.model.visual_visual_learner.parameters(), prompt_params, seen_ids)
            add_unique_params(self.model.text_prompt_photo.parameters(), prompt_params, seen_ids)
            add_unique_params(self.model.text_prompt_sketch.parameters(), prompt_params, seen_ids)
            learner_modules = {'visual_visual_learner', 'text_prompt_photo', 'text_prompt_sketch'}

        ln_params = []
        # Only collect LayerNorms from clip encoders (NOT from learners, already included above)
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                # Skip if inside a learner module (already included with learner params)
                if not any(learner_name in name for learner_name in learner_modules):
                    add_unique_params(module.parameters(recurse=False), ln_params, seen_ids)

        extra_trainable_params = []
        for _, p in self.model.named_parameters():
            if p.requires_grad and id(p) not in seen_ids:
                seen_ids.add(id(p))
                extra_trainable_params.append(p)

        non_prompt_params = ln_params + extra_trainable_params

        prompt_lr = getattr(self.cfg, 'prompt_lr', 1e-5)
        clip_ln_lr = getattr(self.cfg, 'clip_LN_lr', 1e-5)

        param_groups = []
        if prompt_params:
            param_groups.append({'params': prompt_params, 'lr': prompt_lr})
        if non_prompt_params:
            param_groups.append({'params': non_prompt_params, 'lr': clip_ln_lr})

        # Full diagnostic table (replaces the old two "Number of trainable ...
        # params" prints, whose exact numbers are reproduced in section [1]).
        log_param_breakdown(self.model, printer=self.print)

        # No weight_decay (matches ducta/baseline's Adam call, which also omits it -> default 0).
        return torch.optim.Adam(param_groups)

    def _audit_predicted_vs_actual_grads(self):
        """Compare log_param_breakdown's static idle prediction with real grads.

        Silent when they agree. When they disagree it names the offenders, so a
        stale _AUG_FORWARD_BUILDS_GRAPH or a new ablation flag that nobody
        taught _is_idle about surfaces on the first step instead of quietly
        inflating the reported trainable count for a whole run.
        """
        wrong_live, wrong_idle = [], []
        seen = set()
        for name, p in self.model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            group, _sub = _classify_group(name)
            predicted_idle = _is_idle(name, self.cfg, group) is not None
            actually_idle = p.grad is None or p.grad.abs().sum().item() == 0.0
            if predicted_idle and not actually_idle:
                wrong_idle.append((name, p.numel()))
            elif actually_idle and not predicted_idle:
                wrong_live.append((name, p.numel()))

        if not wrong_live and not wrong_idle:
            return
        self.print("WARNING -- param log sai so voi gradient thuc te:")
        for tag, rows in (("log noi LIVE nhung khong co grad", wrong_live),
                          ("log noi IDLE nhung co grad", wrong_idle)):
            if rows:
                self.print(f"    {tag}: {len(rows)} tensors, "
                           f"{sum(c for _, c in rows):,} params")
                for n, c in rows[:10]:
                    self.print(f"        {n}  {c:,}")
                if len(rows) > 10:
                    self.print(f"        ... {len(rows) - 10} more")

    def on_after_backward(self):
        """Diagnostic: does gradient actually reach ctx_photo (layer-0 photo
        prompt)? If grad_norm prints ~0.000000 despite training, this tells us
        whether that's because the parameter never moves (grad ~0 -- real
        graph-disconnection bug) or because it does move but the angular
        drift relative to its own norm is just small at this lr/step count
        (grad non-zero, expected -- not a bug).

        No-op when no_prompt_learning=True -- ctx_photo doesn't exist in that
        mode (no prompts at all). When use_text_visual_exchange=True, checks
        the photo branch's own layer-0 text ctx instead (its closest analogue
        -- visual_visual_learner.ctx_photo doesn't exist in that mode either).

        Also runs a one-shot audit that every parameter log_param_breakdown
        called live really did receive gradient. That log predicts idleness
        statically (from flags and from _AUG_FORWARD_BUILDS_GRAPH) before any
        backward has happened, so this is the only place the prediction can be
        checked against reality.
        """
        if not getattr(self, '_grad_audit_done', False):
            self._grad_audit_done = True
            self._audit_predicted_vs_actual_grads()

        if self.model.no_prompt_learning:
            return
        if self.model.use_text_visual_exchange:
            ctx_photo = self.model.text_visual_learner_photo.ctx
        else:
            ctx_photo = self.model.visual_visual_learner.ctx_photo
        grad_norm = ctx_photo.grad.norm().item() if ctx_photo.grad is not None else 0.0
        param_norm = ctx_photo.detach().norm().item()
        self.log('ctx_photo_grad_norm', grad_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('ctx_photo_param_norm', param_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)

    def training_step(self, batch, batch_idx):
        from src.losses_hicropl import loss_fn_hicropl
        features = self.model(batch, self.classnames)
        # Components are the very tensors summed into `loss` -- reporting only,
        # the returned scalar and its graph are unchanged.
        loss, parts = loss_fn_hicropl(self.args, features, return_components=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        # Per-term epoch curves. Reading a run's outcome needs the aug term's
        # SHARE over time, not just the total: a term that saturates toward 0
        # explains a result differently than a term that never mattered.
        for k, v in parts.items():
            t = v.detach() if torch.is_tensor(v) else torch.tensor(float(v), device=loss.device)
            self.log(k, t, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            # Summed on-device; .item() happens once per epoch, not per step.
            self._loss_parts_sum[k] = self._loss_parts_sum.get(k, 0.0) + t
        self._loss_parts_n += 1

        return loss

    def on_train_epoch_end(self):
        """Print the loss breakdown for THIS epoch, from this module's own sums.

        total is the sum of the three terms by construction, so cross_modal +
        ce + aug always reconciles exactly and aug_share is a share of a number
        that really is the total. It will not match the "Train loss" line
        printed during validation: that one reads an un-suffixed
        callback_metrics key, and for a metric logged with on_step=True that key
        holds the LAST STEP's value, not the epoch mean (validation also runs
        before this epoch's train metrics are reduced). In a converging run the
        last step sits below the epoch mean, so that line reads lower.
        """
        if not self._loss_parts_n:
            return
        n = self._loss_parts_n
        vals = {k: (v / n).item() for k, v in self._loss_parts_sum.items()}
        total = sum(vals.values())
        share = 100.0 * vals.get('loss_aug', 0.0) / total if total else 0.0
        self.print("LOSS_FP | epoch={} | cross_modal={:.6f} | ce={:.6f} | aug={:.6f} "
                   "| total={:.6f} | aug_share={:.2f}% | steps={}".format(
                       self.current_epoch, vals.get('loss_cross_modal', 0.0),
                       vals.get('loss_ce', 0.0), vals.get('loss_aug', 0.0),
                       total, share, n))

    def extract_eval_features(self, tensor, modality):
        """Extract visual features (prompted only, no distill mixing)."""
        if self.model.no_prompt_learning:
            feat = self.model.clip.encode_image(tensor.type(self.model.dtype))
            return feat / feat.norm(dim=-1, keepdim=True)

        if self.model.use_text_visual_exchange:
            learner = (
                self.model.text_visual_learner_photo if modality == 'photo'
                else self.model.text_visual_learner_sketch
            )
            visual_encoder = (
                self.model.visual_encoder_photo if modality == 'photo'
                else self.model.visual_encoder_sketch
            )
            _, vis_shallow, _, vis_deeper = learner()
            feat = visual_encoder(tensor.type(self.model.dtype), vis_shallow, vis_deeper)
            return feat / feat.norm(dim=-1, keepdim=True)

        # Call visual learner once, cache outputs
        photo_shallow, sketch_shallow, photo_deeper, sketch_deeper = self.model.visual_visual_learner()

        if modality == 'photo':
            visual_encoder = self.model.visual_encoder_photo
            vis_shallow, vis_deeper = photo_shallow, photo_deeper
        else:
            visual_encoder = self.model.visual_encoder_sketch
            vis_shallow, vis_deeper = sketch_shallow, sketch_deeper

        feat = visual_encoder(tensor.type(self.model.dtype), vis_shallow, vis_deeper)
        return feat / feat.norm(dim=-1, keepdim=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._validation_step_category(batch, batch_idx, dataloader_idx)

    def _validation_step_category(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 3:
            tensor, label, type_data = batch
        else:
            tensor, label = batch
            type_data = None
            
        if dataloader_idx == 0:
            sketch_feat = self.extract_eval_features(tensor, modality='sketch')
            self.test_sketch_features.append(sketch_feat.cpu().detach()) 
            self.test_sketch_labels.append(label.cpu().detach())
        elif dataloader_idx == 1:
            photo_feat = self.extract_eval_features(tensor, modality='photo')
            self.test_photo_features.append(photo_feat.cpu().detach())   
            self.test_photo_labels.append(label.cpu().detach())

    def on_validation_epoch_end(self):
        return self._on_validation_epoch_end_category()

    def _on_validation_epoch_end_category(self):
        if not self.test_photo_features or not self.test_sketch_features:
            self.print("Warning: Missing features for validation. Skipping metrics.")
            return

        gallery_features = torch.cat(self.test_photo_features, dim=0).to(self.device)
        query_features   = torch.cat(self.test_sketch_features, dim=0).to(self.device)
        
        all_photo_category  = torch.cat(self.test_photo_labels, dim=0).to(self.device)
        all_sketch_category = torch.cat(self.test_sketch_labels, dim=0).to(self.device)

        similarity_matrix = query_features @ gallery_features.t()

        dataset = getattr(self.args, 'dataset', 'sketchy')
        if dataset == "sketchy_2" or dataset == "sketchy_ext":
            map_k = 200
            p_k = 200
        elif dataset == "quickdraw":
            map_k = 0
            p_k = 200
        else:
            map_k = 0
            p_k = 100

        ap = torch.zeros(len(query_features), device=self.device)
        precision = torch.zeros(len(query_features), device=self.device)

        for idx in range(len(query_features)):
            category = all_sketch_category[idx]
            distance = similarity_matrix[idx]
            target = (all_photo_category == category)

            if map_k != 0:
                top_k_actual = min(map_k, len(gallery_features))
                ap[idx] = retrieval_average_precision(distance, target, top_k=top_k_actual)
            else:
                ap[idx] = retrieval_average_precision(distance, target)

            precision[idx] = retrieval_precision(distance, target, top_k=p_k)

        mAP = torch.mean(ap)
        mean_precision = torch.mean(precision)

        self.log("mAP", mAP, on_step=False, on_epoch=True)
        self.log(f"P@{p_k}", mean_precision, on_step=False, on_epoch=True)
        self.log("val_mAP", mAP, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"val_P@{p_k}", mean_precision, on_step=False, on_epoch=True)
        self.log("best_mAP", self.best_metric, on_step=False, on_epoch=True, prog_bar=False)

        if map_k != 0:
            self.log(f"val_map_{map_k}", mAP, on_step=False, on_epoch=True)
        else:
            self.log("val_map_all", mAP, on_step=False, on_epoch=True)
        self.log(f"val_p_{p_k}", mean_precision, on_step=False, on_epoch=True)

        if self.global_step > 0:
            self.best_metric = self.best_metric if (self.best_metric > mAP.item()) else mAP.item()

        if map_k != 0:
            self.print('mAP@{}: {:.4f}, P@{}: {:.4f}, Best mAP: {:.4f}'.format(
                map_k, mAP.item(), p_k, mean_precision.item(), self.best_metric))
        else:
            self.print('mAP@all: {:.4f}, P@{}: {:.4f}, Best mAP: {:.4f}'.format(
                mAP.item(), p_k, mean_precision.item(), self.best_metric))

        train_loss = self.trainer.callback_metrics.get("train_loss", None)
        if train_loss is not None:
            # Label corrected: this key is written on every step, so during
            # validation it holds the LAST STEP of the epoch, not the mean.
            # Measured: it matches step N-1's loss exactly. The epoch mean is
            # the `total` field of the LOSS_FP line printed by
            # on_train_epoch_end.
            self.print(f"Train loss (last step): {train_loss.item():.6f}")

        grad_norm = self.trainer.callback_metrics.get("ctx_photo_grad_norm", None)
        param_norm = self.trainer.callback_metrics.get("ctx_photo_param_norm", None)
        if grad_norm is not None and param_norm is not None:
            self.print(f"[DEBUG] ctx_photo grad_norm (epoch avg): {grad_norm.item():.8f}, param_norm: {param_norm.item():.6f}")

        self.test_photo_features.clear()
        self.test_sketch_features.clear()
        self.test_photo_labels.clear()
        self.test_sketch_labels.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
