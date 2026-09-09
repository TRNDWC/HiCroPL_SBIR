import contextlib
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

# Trainable params clip_aug is supposed to expose. Every LayerNorm is split per
# domain (split_layernorms), so each tower contributes TWICE its single-set size.
#
#   visual, one set: 12 blocks x (ln_1 + ln_2) x (weight + bias) x 768 = 36,864
#                    ln_pre + ln_post          x (weight + bias) x 768 =  3,072
#                                                          single set = 39,936
#   text,   one set: 12 blocks x (ln_1 + ln_2) x (weight + bias) x 512 = 24,576
#                    ln_final                  x (weight + bias) x 512 =  1,024
#                                                          single set = 25,600
#
# The visual tower is always opened; the text tower only when descriptions are
# given, because only then is encode_text actually called on this backbone.
# Verified against real builds. Any other number means a freeze scope moved.
_AUG_LN_VISUAL_EXPECTED = 2 * 39_936          # 79,872
_AUG_LN_TEXT_EXPECTED = 2 * 25_600            # 51,200


def _expected_aug_ln(model):
    """How many trainable LayerNorm params clip_aug should hold in this run.

    The vanilla instance now serves two INDEPENDENT purposes, so each tower is
    counted only when its own condition holds. Under --aug_shared_encoder with
    --text_variant desc_sep the instance exists for the text tower alone, and
    the visual term must not be added.
    """
    total = 0
    if _needs_vanilla_visual(model):
        total += _AUG_LN_VISUAL_EXPECTED
    if _needs_vanilla_text(model):
        total += _AUG_LN_TEXT_EXPECTED
    return total


def _needs_vanilla_visual(model):
    """Is clip_aug's VISUAL tower actually executed this run?"""
    cfg = getattr(model, 'cfg', None)
    return (not getattr(cfg, 'disable_aug_branch', False)
            and not getattr(cfg, 'aug_shared_encoder', False))


def _needs_vanilla_text(model):
    """Is clip_aug's TEXT tower actually executed this run?"""
    return getattr(getattr(model, 'cfg', None), 'text_variant', 'template') == 'desc_sep'


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
            # Canary, one per tower. The vanilla instance serves two independent
            # purposes and __init__ opens each tower only when that tower is
            # really executed, so a trainable param here whose tower is NOT
            # executed means the two conditions have drifted apart.
            if name.startswith('clip_aug.visual.'):
                if not (not getattr(cfg, 'disable_aug_branch', False)
                        and not getattr(cfg, 'aug_shared_encoder', False)):
                    return 'clip_aug visual tower never called (aug views ride the main encoder)'
            elif getattr(cfg, 'text_variant', 'template') != 'desc_sep':
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

    # Run A reference: --aug_shared_encoder means the augmented IMAGE views ride
    # the main encoder, so the vanilla instance must hold no trainable VISUAL
    # LayerNorm. It may still exist for --text_variant desc_sep, whose text
    # tower is a separate concern -- checking the instance's total here would
    # flag that legitimate case as an error.
    if shared_enc:
        n_aug_visual = sum(r['numel'] for r in recs
                           if r['group'] == 'backbone' and r['sub'] == 'aug'
                           and r['name'].startswith('clip_aug.visual.'))
        if n_aug_visual == 0:
            printer(f"    CHECK Run A: clip_aug visual trainable=0 -- OK "
                    f"(instance {'present for the text tower' if clip_aug is not None else 'absent'})")
        else:
            printer(f"    ERROR Run A: expected clip_aug visual trainable=0, got {n_aug_visual}")
    # Reference for the aug backbone: the per-domain LayerNorm sets it is
    # supposed to open -- each tower only when that tower is really executed.
    # See _expected_aug_ln / _needs_vanilla_* for the arithmetic.
    if clip_aug is not None:
        expected = _expected_aug_ln(model)
        towers = '+'.join([t for t, on in (('visual', _needs_vanilla_visual(model)),
                                           ('text', _needs_vanilla_text(model))) if on]) or 'none'
        if n_clip_aug == expected:
            printer(f"    CHECK aug backbone: clip_aug trainable={n_clip_aug:,} -- OK "
                    f"({towers} tower LayerNorm, 2 domains each)")
        else:
            printer(f"    WARNING aug backbone: clip_aug trainable={n_clip_aug:,}, expected "
                    f"{expected:,} (towers executed: {towers}; visual set "
                    f"{_AUG_LN_VISUAL_EXPECTED:,}, text set {_AUG_LN_TEXT_EXPECTED:,})."
                    f" Breakdown by tensor:")
            for r in [x for x in recs if x['group'] == 'backbone' and x['sub'] == 'aug'][:60]:
                printer(f"        {r['name']}  {r['numel']:,}  "
                        f"{'LN' if r['is_ln'] else 'NOT-LN'}")
    printer("=" * 78)
    printer("")




def log_desc_fingerprint(cfg, n_cls, sha_s, sha_p, eot_ce=None, eot_aux=None,
                         aux_encoder='none', printer=print):
    """One greppable line describing the text branch of this run.

    eot_ce / eot_aux are tokenized_prompts.argmax(-1) of the two sequences --
    the position TextEncoder actually pools at, i.e. the number that silently
    goes wrong when the three co-dependent buffers drift apart.
    """
    import statistics

    def stats(v):
        if v is None or len(v) == 0:
            return 'n/a', 'n/a', 'n/a'
        vals = [int(x) for x in v]
        return min(vals), int(round(statistics.median(vals))), max(vals)

    c_min, c_med, c_max = stats(eot_ce)
    a_min, a_med, a_max = stats(eot_aux)
    printer(f"DESC_FP | variant={getattr(cfg, 'text_variant', 'template')} | "
            f"lambda_text={getattr(cfg, 'lambda_text', 1.0):.3f} | n_cls={n_cls} | "
            f"n_ctx={getattr(cfg, 'n_ctx', 4)} | sha_s={sha_s} | sha_p={sha_p} | "
            f"eot_ce_min={c_min} | eot_ce_med={c_med} | eot_ce_max={c_max} | "
            f"eot_aux_min={a_min} | eot_aux_med={a_med} | eot_aux_max={a_max} | "
            f"aux_encoder={aux_encoder}")




class DomainLayerNorm(nn.Module):
    """Two LayerNorms in one slot, selected by a process-wide active domain.

    Gives sketch and photo fully disjoint TRAINABLE backbone parameters. Since
    LayerNorm is the only trainable part of the frozen CLIP backbone, routing it
    per domain is numerically identical to keeping two complete encoder copies,
    while duplicating 0.25 MiB instead of 577 MiB: the other 151M weights are
    frozen, identical in both copies, and never updated, so sharing the objects
    cannot change any output.

    Done without touching src/clip/model.py: the CLIP blocks keep calling
    `self.ln_1(x)` / `self.ln_final(x)` with a single argument, and the routing
    happens inside.

    Both copies start from the SAME pretrained tensor (deepcopy, not a fresh
    init), so at step 0 the split is numerically invisible -- any difference that
    appears later comes from training, not from initialisation.

    The active domain is a CLASS attribute, not per-instance state: one forward
    pass touches 51 of these modules, and setting a flag on each would be 51
    chances to miss one.

    `weight` / `bias` / `normalized_shape` / `eps` proxy to the active copy so
    read-only introspection keeps working -- src/hicropl.py:188 and :852 size the
    text context with `clip_model.ln_final.weight.shape[0]`, which would raise
    AttributeError otherwise. Both copies always share a shape, so which one
    answers does not matter.
    """

    _ACTIVE = 'photo'

    def __init__(self, layer_norm):
        super().__init__()
        self.photo = copy.deepcopy(layer_norm)
        self.sketch = copy.deepcopy(layer_norm)

    def forward(self, x):
        return self.active(x)

    @property
    def active(self):
        return self.photo if DomainLayerNorm._ACTIVE == 'photo' else self.sketch

    @property
    def weight(self):
        return self.active.weight

    @property
    def bias(self):
        return self.active.bias

    @property
    def normalized_shape(self):
        return self.active.normalized_shape

    @property
    def eps(self):
        return self.active.eps

    def extra_repr(self):
        return f"active={DomainLayerNorm._ACTIVE}"


@contextlib.contextmanager
def active_domain(domain):
    """Route every DomainLayerNorm to `domain` for the duration of the block.

    Numerically inert when no DomainLayerNorm is in the graph, because no
    DomainLayerNorm exists then. Restores the previous value on exit so nested
    or interleaved use cannot leak state into the next call.
    """
    if domain not in ('photo', 'sketch'):
        raise ValueError(f"domain must be 'photo' or 'sketch', got {domain!r}")
    previous = DomainLayerNorm._ACTIVE
    DomainLayerNorm._ACTIVE = domain
    try:
        yield
    finally:
        DomainLayerNorm._ACTIVE = previous


def split_layernorms(root):
    """Replace every nn.LayerNorm under `root` with a DomainLayerNorm.

    Returns the number of MODULES replaced. For a whole ViT-B/32 CLIP that is 51
    (26 visual: 12 blocks x ln_1/ln_2 + ln_pre + ln_post; 25 text: 12 blocks x
    ln_1/ln_2 + ln_final) -- 102 tensors before the split, 204 after.

    Called BEFORE freeze_all_but_bn on purpose: that function walks
    model.modules(), which recurses into the two children of each
    DomainLayerNorm, so both copies still get unfrozen. Wrapping them does not
    hide them from the freeze logic, from configure_optimizers' LayerNorm sweep,
    or from log_param_breakdown -- all three test isinstance(m, nn.LayerNorm)
    over a recursive walk.
    """
    replaced = 0
    for module in list(root.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.LayerNorm):
                setattr(module, child_name, DomainLayerNorm(child))
                replaced += 1
    return replaced


def log_dsln_fingerprint(model, n_replaced, printer=print):
    """One greppable line: how the visual LayerNorm budget ended up split."""
    vis_ids = {id(p) for p in model.clip.visual.parameters()}
    ln_ids = {id(p) for m in model.modules() if isinstance(m, nn.LayerNorm)
              for p in m.parameters()}
    seen, ln_v, ln_t, total = set(), 0, 0, 0
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if id(p) in ln_ids and name.startswith('clip.'):
            if id(p) in vis_ids:
                ln_v += p.numel()
            else:
                ln_t += p.numel()
    aug = getattr(model, 'clip_aug', None)
    ln_aug = 0 if aug is None else sum(
        p.numel() for p in aug.parameters() if p.requires_grad and id(p) in ln_ids)
    printer(f"DSLN_FP | n_ln_split_main={n_replaced} | n_ln_split_aug="
            f"{getattr(model, '_n_ln_replaced_aug', 0)} | ln_visual_params={ln_v} | "
            f"ln_text_params={ln_t} | ln_aug_params={ln_aug} | total_trainable={total}")


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
        # Per-branch LayerNorm across the WHOLE backbone -- visual tower AND
        # text tower. Always on: sketch and photo share no trainable backbone
        # parameter, which is what "separate encoders" reduces to here, since
        # LayerNorm is the only trainable part of a frozen CLIP.
        #
        # Done BEFORE freeze_all_but_bn so both copies land in its LayerNorm
        # sweep, and before the TextEncoder wrappers are built, because
        # TextEncoder captures `clip_model.ln_final` by reference at
        # construction time (src/hicropl.py:96) -- splitting afterwards would
        # leave it pointing at the discarded original. deepcopy consumes no RNG,
        # so every module built after this point keeps its stream.
        self._n_ln_replaced = split_layernorms(self.clip)
        freeze_all_but_bn(self.clip)
        # Param counts are reported once by log_param_breakdown() in
        # configure_optimizers -- the single source of truth.

        # Single shared logit scale (matches ducta/baseline)
        self.logit_scale = self.clip.logit_scale

        # -- Class descriptions (--desc_sketch/--desc_photo), both or neither --
        # Loaded here, before any learner is built, so a coverage gap fails now
        # rather than after a class silently falls back to a different prompt
        # shape than its neighbours.
        desc_sketch = desc_photo = None
        sha_s = sha_p = 'n/a'
        desc_pos = getattr(cfg, 'desc_pos', 'V1')
        path_s = getattr(cfg, 'desc_sketch', None)
        path_p = getattr(cfg, 'desc_photo', None)
        if (path_s is None) != (path_p is None):
            raise ValueError(
                "desc_sketch and desc_photo must be given together or not at all; got "
                f"desc_sketch={path_s!r}, desc_photo={path_p!r}"
            )
        if path_s is not None:
            from src.utils import load_class_descriptions
            desc_sketch, sha_s = load_class_descriptions(path_s)
            desc_photo, sha_p = load_class_descriptions(path_p)
            # P6: every class the model classifies over must have a description
            # in BOTH files, otherwise the CE logits mix two prompt formats.
            missing_s = sorted(set(classnames) - set(desc_sketch))
            missing_p = sorted(set(classnames) - set(desc_photo))
            if missing_s or missing_p:
                raise ValueError(
                    f"Class descriptions do not cover the model's classes. "
                    f"|classnames|={len(classnames)}, |desc_sketch|={len(desc_sketch)}, "
                    f"|desc_photo|={len(desc_photo)}. "
                    f"Missing from sketch ({len(missing_s)}): {missing_s[:10]}"
                    f"{' ...' if len(missing_s) > 10 else ''}. "
                    f"Missing from photo ({len(missing_p)}): {missing_p[:10]}"
                    f"{' ...' if len(missing_p) > 10 else ''}."
                )
        # Text-branch architecture. Deliberately independent of every aug flag:
        # the vanilla-instance build condition further down ORs this in rather
        # than reading --aug_shared_encoder.
        self.text_variant = getattr(cfg, 'text_variant', 'template')
        self.lambda_text = getattr(cfg, 'lambda_text', 1.0)
        if self.text_variant != 'template' and desc_sketch is None:
            raise ValueError(
                f"--text_variant {self.text_variant} needs --desc_sketch and --desc_photo"
            )
        self._has_descriptions = desc_sketch is not None

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
            self.text_prompt_photo = SimpleTextPromptLearner(
                cfg_photo, classnames, self.clip, descriptions=desc_photo, desc_pos=desc_pos,
                text_variant=self.text_variant)

            print("Initializing Sketch Text Prompt Learner...")
            cfg_sketch = copy.copy(cfg)
            cfg_sketch.ctx_init = getattr(cfg, 'ctx_init_sketch', 'a sketch of a')
            self.text_prompt_sketch = SimpleTextPromptLearner(
                cfg_sketch, classnames, self.clip, descriptions=desc_sketch, desc_pos=desc_pos,
                text_variant=self.text_variant)

            _L = self.text_prompt_sketch
            _aux_tok = getattr(_L, 'tokenized_prompts_aux', None)
            log_desc_fingerprint(
                cfg, len(classnames), sha_s, sha_p,
                eot_ce=_L.tokenized_prompts.argmax(dim=-1),
                eot_aux=None if _aux_tok is None else _aux_tok.argmax(dim=-1),
                aux_encoder=({'desc_sep': 'vanilla', 'desc_shared': 'main'}
                             .get(self.text_variant, 'none')))

            # -- Encoders (both branches wrap the SAME shared backbone) --
            self.text_encoder_photo = TextEncoder(self.clip)
            self.text_encoder_sketch = TextEncoder(self.clip)
            self.visual_encoder_photo = VisualEncoder(self.clip)
            self.visual_encoder_sketch = VisualEncoder(self.clip)

        # The default branch already logged DESC_FP with real eot_* values; the
        # other two architectures have no SimpleTextPromptLearner to read them
        # from, so they log the same line with eot_*=n/a.
        if self.no_prompt_learning or self.use_text_visual_exchange:
            log_desc_fingerprint(cfg, len(classnames), sha_s, sha_p)
            if desc_sketch is not None:
                print("[WARN] class descriptions were loaded but this architecture branch does "
                      "not use SimpleTextPromptLearner, so they are IGNORED "
                      f"(no_prompt_learning={self.no_prompt_learning}, "
                      f"use_text_visual_exchange={self.use_text_visual_exchange}).")

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
        self._n_ln_replaced_aug = 0
        # ONE vanilla instance serves two independent purposes, so the decision
        # to build it is the OR of two independent conditions -- the text branch
        # must not depend on an image-branch flag:
        #   visual tower -> prompt-free encoder for the augmented image views
        #   text tower   -> prompt-free encoder for the description sequence
        # `self.clip_aug` stays the attribute name so every existing call site,
        # log line and param-group rule keeps working unchanged.
        need_vanilla_visual = not self.disable_aug_branch and not self.aug_shared_encoder
        need_vanilla_text = self.text_variant == 'desc_sep'
        if need_vanilla_visual or need_vanilla_text:
            from src.utils import load_clip_to_cpu
            cfg_aug = copy.copy(cfg)
            cfg_aug.clip_trainer = 'CoOp'
            # vision_depth=0 makes this backbone genuinely prompt-free.
            #
            # Without it, the CoOp build's VisionTransformer creates
            # self.VPT = nn.Parameter(normal_(std=0.02)) whenever vision_depth
            # != 0 (src/clip/model.py:441-452) and its forward CONCATENATES
            # those tokens onto every image (src/clip/model.py:479-481). CLIP's
            # pretrained state_dict has no such key, so they stay at their random
            # init -- and freeze_model() then locks them there for the whole run.
            # Measured effect: cos(feature with those tokens, feature without)
            # = 0.929, i.e. the "vanilla reference" was neither vanilla nor
            # prompt-free. The stray key is also what printed
            # "Weights not found for some missing keys: ['visual.VPT']".
            #
            # With 0, VPT_shallow is False, no VPT parameter is built, and
            # forward takes the `else` branch whose `assert
            # self.prompt_till_layer_visual == 0` holds because
            # prompt_till_layer_visual is set from this same value
            # (src/clip/model.py:460). The text tower is unaffected: the CoOp
            # build uses plain ResidualAttentionBlock either way.
            cfg_aug.vision_depth = 0
            self.clip_aug = load_clip_to_cpu(cfg_aug).to(original_device)
            # Per-domain split on BOTH towers of the aug backbone: photo_aug and
            # sketch_aug must not share LayerNorm, and neither must the photo and
            # sketch description prompts when they run through this text tower.
            self._n_ln_replaced_aug = split_layernorms(self.clip_aug)
            # Freeze everything, then reopen LayerNorm only where this branch
            # actually computes something.
            #
            # freeze_all_but_bn(self.clip_aug) would be the obvious call, but it
            # opens LayerNorm in both towers unconditionally. A tower that is
            # never called would then sit in the optimizer with .grad = None for
            # the entire run: real dead weight, and an IDLE row in the param log
            # of every experiment. Each tower is opened only when something
            # actually runs through it, which keeps declared == effective.
            freeze_model(self.clip_aug)
            if need_vanilla_visual:
                self.clip_aug.visual.apply(unfreeze_ln)
            if need_vanilla_text:
                self.clip_aug.transformer.apply(unfreeze_ln)
                self.clip_aug.ln_final.apply(unfreeze_ln)
            self.clip_aug.eval()

        log_dsln_fingerprint(self, self._n_ln_replaced)

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
        # Description text features -- stay None unless --desc_sketch/--desc_photo
        # were given AND this architecture branch has a SimpleTextPromptLearner.
        text_features_desc_photo = text_features_desc_sketch = None
        run_shared_aug = self.aug_shared_encoder and photo_aug_tensor is not None

        if self.no_prompt_learning:
            # Plain frozen CLIP forward (only LayerNorm trainable) -- no
            # prompt tensors of any kind, text uses the fixed template.
            with active_domain('photo'):
                image_features_photo = self.clip.encode_image(photo_tensor.type(self.dtype))
            with active_domain('sketch'):
                image_features_sketch = self.clip.encode_image(sk_tensor.type(self.dtype))
            # neg_tensor is a PHOTO of a different category (src/dataset_retrieval.py:131),
            # so it uses the photo LayerNorms -- not the sketch ones.
            with active_domain('photo'):
                image_features_neg = self.clip.encode_image(neg_tensor.type(self.dtype))
            with active_domain('photo'):
                text_features_all_photo = self.clip.encode_text(self.tokenized_prompts_photo)
            with active_domain('sketch'):
                text_features_all_sketch = self.clip.encode_text(self.tokenized_prompts_sketch)
            if run_shared_aug:
                with active_domain('photo'):
                    image_features_photo_aug = self.clip.encode_image(photo_aug_tensor.type(self.dtype))
                with active_domain('sketch'):
                    image_features_sketch_aug = self.clip.encode_image(sk_aug_tensor.type(self.dtype))
        elif self.use_text_visual_exchange:
            # Each branch's learner performs its OWN bidirectional text<->visual
            # exchange -- no coupling between the two learners/branches.
            text_input_photo_all, vis_shallow_photo, cross_prompts_text_deeper_photo, vis_deeper_photo = self.text_visual_learner_photo()
            with active_domain('photo'):
                text_features_all_photo = self.text_encoder_photo(text_input_photo_all, self.text_visual_learner_photo.tokenized_prompts, cross_prompts_text_deeper_photo)
            with active_domain('photo'):
                image_features_photo = self.visual_encoder_photo(photo_tensor.type(self.dtype), vis_shallow_photo, vis_deeper_photo)

            text_input_sketch_all, vis_shallow_sketch, cross_prompts_text_deeper_sketch, vis_deeper_sketch = self.text_visual_learner_sketch()
            with active_domain('sketch'):
                text_features_all_sketch = self.text_encoder_sketch(text_input_sketch_all, self.text_visual_learner_sketch.tokenized_prompts, cross_prompts_text_deeper_sketch)
            with active_domain('sketch'):
                image_features_sketch = self.visual_encoder_sketch(sk_tensor.type(self.dtype), vis_shallow_sketch, vis_deeper_sketch)

            # Negative branch (uses photo encoder + photo visual prompts)
            with active_domain('photo'):   # neg_tensor is a photo
                image_features_neg = self.visual_encoder_photo(neg_tensor.type(self.dtype), vis_shallow_photo, vis_deeper_photo)

            if run_shared_aug:
                # Same encoders, same prompt tensors as the clean views above.
                with active_domain('photo'):
                    image_features_photo_aug = self.visual_encoder_photo(photo_aug_tensor.type(self.dtype), vis_shallow_photo, vis_deeper_photo)
                with active_domain('sketch'):
                    image_features_sketch_aug = self.visual_encoder_sketch(sk_aug_tensor.type(self.dtype), vis_shallow_sketch, vis_deeper_sketch)
        else:
            # 1. Call visual-visual learner ONCE (shared by both branches)
            photo_shallow, sketch_shallow, photo_deeper, sketch_deeper = self.visual_visual_learner()

            # 2. Photo branch: text learner + visual routing
            # Compute text features for ALL classes (not just batch) - needed for loss computation
            text_input_photo_all, cross_prompts_text_deeper_photo = self.text_prompt_photo(label=None)  # All classes
            with active_domain('photo'):
                text_features_all_photo = self.text_encoder_photo(text_input_photo_all, self.text_prompt_photo.tokenized_prompts, cross_prompts_text_deeper_photo)
                # Description branch, shared-encoder variant: SAME encoder, SAME
                # ctx, SAME deep prompts (cross_prompts_text_deeper_photo is
                # reused, the learner is not called again) -- only the frozen
                # token content differs. Exact text-side mirror of what
                # --aug_shared_encoder does to the augmented image view.
                # desc_shared: the auxiliary sequence rides THIS encoder with the
                # same ctx and the same deep prompts (the learner is not called
                # again). Gated only by --text_variant, never by an aug flag.
                aux_input_photo = self.text_prompt_photo.forward_aux()
                if aux_input_photo is not None:
                    text_features_desc_photo = self.text_encoder_photo(aux_input_photo, self.text_prompt_photo.tokenized_prompts_aux, cross_prompts_text_deeper_photo)
            with active_domain('photo'):
                image_features_photo = self.visual_encoder_photo(photo_tensor.type(self.dtype), photo_shallow, photo_deeper)

            # 3. Sketch branch: text learner + visual routing
            # Compute text features for ALL classes (not just batch) - needed for loss computation
            text_input_sketch_all, cross_prompts_text_deeper_sketch = self.text_prompt_sketch(label=None)  # All classes
            with active_domain('sketch'):
                text_features_all_sketch = self.text_encoder_sketch(text_input_sketch_all, self.text_prompt_sketch.tokenized_prompts, cross_prompts_text_deeper_sketch)
                aux_input_sketch = self.text_prompt_sketch.forward_aux()
                if aux_input_sketch is not None:
                    text_features_desc_sketch = self.text_encoder_sketch(aux_input_sketch, self.text_prompt_sketch.tokenized_prompts_aux, cross_prompts_text_deeper_sketch)
            with active_domain('sketch'):
                image_features_sketch = self.visual_encoder_sketch(sk_tensor.type(self.dtype), sketch_shallow, sketch_deeper)

            # 4. Negative branch (uses photo encoder + photo visual prompts)
            with active_domain('photo'):   # neg_tensor is a photo
                image_features_neg = self.visual_encoder_photo(neg_tensor.type(self.dtype), photo_shallow, photo_deeper)

            # 4b. Run A: augmented views, SAME encoders and SAME prompt tensors
            # (photo_shallow/photo_deeper, sketch_shallow/sketch_deeper) as
            # steps 2-3 -- the learner is not called a second time, so the only
            # thing that differs from the clean pass is the input tensor.
            if run_shared_aug:
                with active_domain('photo'):
                    image_features_photo_aug = self.visual_encoder_photo(photo_aug_tensor.type(self.dtype), photo_shallow, photo_deeper)
                with active_domain('sketch'):
                    image_features_sketch_aug = self.visual_encoder_sketch(sk_aug_tensor.type(self.dtype), sketch_shallow, sketch_deeper)

        # 5. Normalize features
        photo_feat = image_features_photo / image_features_photo.norm(dim=-1, keepdim=True)
        sketch_feat = image_features_sketch / image_features_sketch.norm(dim=-1, keepdim=True)
        neg_feat = image_features_neg / image_features_neg.norm(dim=-1, keepdim=True)
        text_feat_photo = text_features_all_photo / text_features_all_photo.norm(dim=-1, keepdim=True)
        text_feat_sketch = text_features_all_sketch / text_features_all_sketch.norm(dim=-1, keepdim=True)
        text_desc_feat_photo = text_desc_feat_sketch = None
        if text_features_desc_photo is not None:
            text_desc_feat_photo = text_features_desc_photo / text_features_desc_photo.norm(dim=-1, keepdim=True)
            text_desc_feat_sketch = text_features_desc_sketch / text_features_desc_sketch.norm(dim=-1, keepdim=True)

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
        # `self.clip_aug is not None` used to be a proxy for "the aug branch is
        # on", back when that instance existed for no other reason. It can now
        # be built purely for --text_variant desc_sep, so the aug branch must be
        # gated on its OWN flag. Without this, feeding augmented tensors while
        # --disable_aug_branch is set would silently revive loss_aug.
        elif (not self.disable_aug_branch and photo_aug_tensor is not None
                and self.clip_aug is not None):
            with active_domain('photo'):
                f_p = self.clip_aug.encode_image(photo_aug_tensor.type(self.dtype))
            with active_domain('sketch'):
                f_s = self.clip_aug.encode_image(sk_aug_tensor.type(self.dtype))
            photo_aug_feat = f_p / f_p.norm(dim=-1, keepdim=True)
            sketch_aug_feat = f_s / f_s.norm(dim=-1, keepdim=True)

        # 7b. Description branch, second-encoder variant -- the text-side twin of
        # the block just above. clip_aug already carries a full text tower that
        # was previously never called, so this costs no extra weights: the
        # descriptions run through it prompt-free via encode_text(), exactly as
        # the augmented image runs through clip_aug.visual prompt-free.
        #
        # Uses tokenized_prompts_desc_plain, NOT tokenized_prompts_desc: there is
        # no ctx here to overwrite the "X" window, so the placeholders must not be
        # in the sequence at all.
        _tp = getattr(self, 'text_prompt_photo', None)   # absent in the other two architectures
        if (self.text_variant == 'desc_sep' and self.clip_aug is not None
                and _tp is not None and _tp.has_aux):
            with active_domain('photo'):
                text_features_desc_photo = self.clip_aug.encode_text(
                    self.text_prompt_photo.tokenized_prompts_aux)
            with active_domain('sketch'):
                text_features_desc_sketch = self.clip_aug.encode_text(
                    self.text_prompt_sketch.tokenized_prompts_aux)
            text_desc_feat_photo = text_features_desc_photo / text_features_desc_photo.norm(dim=-1, keepdim=True)
            text_desc_feat_sketch = text_features_desc_sketch / text_features_desc_sketch.norm(dim=-1, keepdim=True)

        # Snapshot for TEXT_FP. Text features depend only on the prompts, not on
        # the image batch, so the last step of an epoch describes that epoch's
        # end state exactly. Detached: reporting must not touch the graph.
        self._text_snapshot = (
            text_feat_photo.detach(), text_feat_sketch.detach(),
            None if text_desc_feat_photo is None else text_desc_feat_photo.detach(),
            None if text_desc_feat_sketch is None else text_desc_feat_sketch.detach(),
        )

        return (
            photo_feat, logits_photo,
            sketch_feat, logits_sketch,
            neg_feat, label,
            text_feat_photo, text_feat_sketch,
            photo_aug_feat, sketch_aug_feat,
            text_desc_feat_photo, text_desc_feat_sketch,
        )


class HiCroPL_SBIR(pl.LightningModule):
    def __init__(self, cfg, args, classnames, model):
        super().__init__()
        self.cfg = cfg
        self.args = args
        self.classnames = classnames
        self.model = model
        
        self.best_metric = 1e-3
        # Companions to best_metric: P@k and the epoch measured AT THE SAME
        # validation pass, so the summary file reports a matched pair instead of
        # max(mAP) next to an unrelated P@k.
        self.best_precision = 0.0
        self.best_epoch = -1
        self.best_metric_name = ''
        self.best_precision_name = ''
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
        # ||d(loss_aug)/d(photo_feat)||, sampled at the first and last step of
        # the epoch. None = the aug term is not a tensor this run (branch off).
        self._aug_grad_first = None
        self._aug_grad_last = None

    def on_fit_start(self):
        """Records the wall-clock start so on_fit_end can report a duration.

        Was intentionally empty before.

        Used to print per-branch learnable-token counts; that information is now
        covered (in params, not token counts) by log_param_breakdown() in
        configure_optimizers, which is the single source of truth.
        """
        import time
        self._fit_started_at = time.time()

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

        # Life-or-death check for any run whose aug term is degenerate: does
        # loss_aug still reach photo_feat at all? A saturated positive term with
        # a live negative term still has non-zero gradient; a hidden detach has
        # exactly zero. Only the total loss is observable otherwise, and those
        # two cases are indistinguishable in it.
        n_batches = self.trainer.num_training_batches
        is_last = isinstance(n_batches, int) and batch_idx == n_batches - 1
        if batch_idx == 0 or is_last:
            self._probe_aug_grad(parts.get('loss_aug'), features[0],
                                 first=(batch_idx == 0))

        return loss

    def _print_text_fp(self):
        """TEXT_FP: how far apart the two text sequences actually are.

        cos_TA_*    -- mean over classes of cos(T_c, A_c): how tightly loss_text
                       has pulled the L_ce sequence and the description together.
        offdiag_A_* -- mean over class PAIRS of cos(A_c, A_c'): how spread the
                       description features are. A collapsing auxiliary branch
                       shows up here before it shows up in mAP.
        gap_TT      -- mean over classes of cos(T_c^photo, T_c^sketch): the
                       photo/sketch text gap, unchanged by this branch at init.

        Printed only for the variants that actually have an auxiliary sequence.
        """
        snap = getattr(self.model, '_text_snapshot', None)
        if snap is None or snap[2] is None:
            return
        t_p, t_s, a_p, a_s = snap

        def offdiag(x):
            sim = x @ x.t()
            n = sim.shape[0]
            if n < 2:
                return float('nan')
            return ((sim.sum() - sim.diag().sum()) / (n * (n - 1))).item()

        self.print("TEXT_FP | ep={} | cos_TA_photo={:.4f} | cos_TA_sketch={:.4f} "
                   "| offdiag_A_photo={:.4f} | offdiag_A_sketch={:.4f} "
                   "| gap_TT={:.4f}".format(
                       self.current_epoch,
                       (t_p * a_p).sum(-1).mean().item(),
                       (t_s * a_s).sum(-1).mean().item(),
                       offdiag(a_p), offdiag(a_s),
                       (t_p * t_s).sum(-1).mean().item()))

    def _probe_aug_grad(self, loss_aug, photo_feat, first):
        """d(loss_aug)/d(photo_feat), L2 norm -- read-only.

        torch.autograd.grad with an explicit `inputs` returns the gradient
        instead of accumulating it into .grad, and retain_graph=True leaves the
        graph intact for the real backward that Lightning runs afterwards. No
        second backward(), no optimizer interaction, no RNG consumed -- verified
        by comparing post-fit parameter checksums with and without this probe.
        """
        if not torch.is_tensor(loss_aug) or not loss_aug.requires_grad:
            value = None
        else:
            g = torch.autograd.grad(loss_aug, photo_feat, retain_graph=True,
                                    allow_unused=True)[0]
            value = 0.0 if g is None else g.norm().item()
        if first:
            self._aug_grad_first = value
        else:
            self._aug_grad_last = value

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
        fmt = lambda x: 'n/a' if x is None else '{:.6e}'.format(x)
        self.print("LOSS_FP | ep={} | cross_modal={:.6f} | ce={:.6f} | aug={:.6f} "
                   "| text={:.6f} | total={:.6f} | aug_grad_norm={} "
                   "| aug_grad_norm_first={} | aug_share={:.2f}% | steps={}".format(
                       self.current_epoch, vals.get('loss_cross_modal', 0.0),
                       vals.get('loss_ce', 0.0), vals.get('loss_aug', 0.0),
                       vals.get('loss_text', 0.0), total,
                       fmt(self._aug_grad_last), fmt(self._aug_grad_first), share, n))
        self._print_text_fp()
        # Same four values as scalars, plus the probe.
        self.log('loss_total', torch.tensor(float(total), device=self.device),
                 on_step=False, on_epoch=True, logger=True)
        if self._aug_grad_last is not None:
            self.log('aug_grad_norm', torch.tensor(float(self._aug_grad_last), device=self.device),
                     on_step=False, on_epoch=True, logger=True)

    def on_fit_end(self):
        """Write the best result of this run to results/<exp_name>.txt.

        Two outputs, both append-safe: one file per experiment (easy to open for
        a single run) and one shared CSV row (easy to sort when filling a table
        of 20+ runs). Metric names are carried through rather than hard-coded,
        because the metric depends on the dataset -- mAP@200/P@200 for
        sketchy_ext, mAP@all/P@100 for tuberlin, mAP@all/P@200 for quickdraw
        (see _on_validation_epoch_end_category).
        """
        import csv
        import os
        import shlex
        import sys
        import time

        if not self.trainer.is_global_zero:
            return
        finished_at = time.strftime('%Y-%m-%d %H:%M:%S')

        # Reconstruct the command in `python -m <module>` form rather than
        # joining sys.argv verbatim: sys.argv[0] is the script PATH, and running
        # that path directly puts experiments/ on sys.path instead of the repo
        # root, so `from src...` would fail. The -m form is what actually reruns.
        try:
            module = os.path.splitext(os.path.relpath(sys.argv[0], os.getcwd()))[0]
            module = module.replace(os.sep, '.')
            launcher = f"python -m {module}"
        except ValueError:                      # different drive on Windows
            launcher = f"python {sys.argv[0]}"
        command = " ".join([launcher] + [shlex.quote(a) for a in sys.argv[1:]])
        started = getattr(self, '_fit_started_at', None)
        elapsed = '' if started is None else time.strftime('%H:%M:%S', time.gmtime(time.time() - started))
        exp = getattr(self.args, 'exp_name', 'run')
        out_dir = 'results'
        os.makedirs(out_dir, exist_ok=True)

        # The flags that actually distinguish one ablation cell from another.
        flag_names = ('dataset', 'epochs', 'n_ctx', 'prompt_depth', 'cross_layer',
                      'prompt_lr', 'clip_LN_lr', 'disable_exchange',
                      'exchange_detach_source', 'disable_aug_branch',
                      'aug_shared_encoder', 'aug_identity_transform',
                      'allow_degenerate_aug', 'text_variant')
        flags = {k: getattr(self.args, k, None) for k in flag_names}

        m_name = self.best_metric_name or 'best_metric'
        p_name = self.best_precision_name or 'P'
        # APPEND, never truncate: re-running the same --exp_name must add a new
        # entry rather than destroy the previous one, so a repeated or resumed
        # run can be compared against its predecessor. The timestamp is what
        # tells the entries apart.
        lines = ["=" * 66,
                 f"finished_at = {finished_at}" + (f"   (elapsed {elapsed})" if elapsed else ""),
                 f"exp_name   = {exp}",
                 f"best_epoch = {self.best_epoch}",
                 f"{m_name:<10} = {self.best_metric:.4f}",
                 f"{p_name:<10} = {self.best_precision:.4f}",
                 f"epochs_run = {self.current_epoch}",
                 f"cwd        = {os.getcwd()}",
                 "flags:"]
        lines += [f"    {k} = {v}" for k, v in flags.items()]
        lines += ["command:", f"    {command}"]
        with open(os.path.join(out_dir, f"{exp}.txt"), 'a') as f:
            f.write("\n".join(lines) + "\n")

        # csv.writer, not manual f-string joining: the command field can contain
        # commas or quotes and would otherwise split into bogus columns.
        csv_path = os.path.join(out_dir, 'summary.csv')
        header = ["finished_at", "elapsed", "exp_name", "dataset", "metric", "best_value",
                  "precision_name", "precision", "best_epoch", "command"]
        row = [finished_at, elapsed, exp, flags['dataset'], m_name, f"{self.best_metric:.4f}",
               p_name, f"{self.best_precision:.4f}", self.best_epoch, command]
        need_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(header)
            writer.writerow(row)

        self.print(f"BEST_FP | exp={exp} | {m_name}={self.best_metric:.4f} | "
                   f"{p_name}={self.best_precision:.4f} | epoch={self.best_epoch} | "
                   f"at={finished_at} | elapsed={elapsed or 'n/a'} | "
                   f"-> {out_dir}/{exp}.txt, {csv_path}")

    def extract_eval_features(self, tensor, modality):
        """Extract visual features (prompted only, no distill mixing)."""
        # modality is the ground truth for which LayerNorm set to use at eval:
        # sketch queries -> sketch LN, photo gallery -> photo LN. Getting this
        # wrong lowers mAP without raising anything, so the whole body runs
        # inside the context rather than only the encoder call.
        with active_domain('photo' if modality == 'photo' else 'sketch'):
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
        if getattr(self.args, 'cross_dataset_eval', False):
            # Across-dataset ZS-SBIR always reports mAP@all, P@100 regardless
            # of which target dataset (tuberlin/quickdraw) is being evaluated --
            # overrides the per-dataset map_k/p_k convention below.
            map_k = 0
            p_k = 100
        elif dataset == "sketchy_2" or dataset == "sketchy_ext":
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

        if self.global_step > 0 and mAP.item() >= self.best_metric:
            # Same update rule as before (the old expression kept the old value
            # only when best > mAP, i.e. it replaced on >=); it just records the
            # companions now.
            self.best_metric = mAP.item()
            self.best_precision = mean_precision.item()
            self.best_epoch = self.current_epoch
            self.best_metric_name = f'mAP@{map_k}' if map_k != 0 else 'mAP@all'
            self.best_precision_name = f'P@{p_k}'

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
