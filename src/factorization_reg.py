"""
Modality-semantic factorization regularizer for prompt-tuned CLIP text branches
(ZS-SBIR). Ported from the reference implementation supplied by the project
advisor (factorization_reg_plugin.py); `estimate_subspaces`/`FactorizationReg`
below are kept faithful to that reference (already verified there) -- only the
CLIP text-encoding boundary (`build_for_sbir`) is adapted to this codebase's
two-branch (clip_photo/clip_sketch) architecture.

Adds two text-branch losses on top of the existing objective:
    L_leak = || P_S delta ||^2       protect the class-semantic subspace S
                                      (both branches)
    L_par  = || (I - P_M) g ||^2     keep the sketch/photo gap on the shared
                                      modality axis M (normalized by ||g||^2
                                      by default -> scale-free in [0,1])

M/S are estimated from a BROAD external vocabulary (ImageNet-1k classnames),
not the ~100 seen SBIR classnames. This is the key fix over an earlier
attempt (TextSubspaceRegularizer, now removed) that estimated S directly from
the seen classnames -- that made S span almost the same space L_cls needs to
separate those classes, a direct optimization conflict confirmed empirically
(-1.03 mAP). A broad, disjoint vocabulary keeps S representing general
"category meaning" rather than "how these particular seen classes differ",
leaving room for L_cls's own job.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------
# core subspace estimation (verbatim from the advisor's reference implementation)
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def estimate_subspaces(feat_photo, feat_sketch, r_m=2, s_var=0.90, tol=1e-6):
    delta = feat_sketch - feat_photo
    _, sM, VhM = torch.linalg.svd(delta, full_matrices=False)
    M = VhM[:r_m].contiguous()
    evr_M = (sM ** 2) / (sM ** 2).sum()

    abar = 0.5 * (feat_photo + feat_sketch)
    abar = abar - abar.mean(0, keepdim=True)
    _, sS, VhS = torch.linalg.svd(abar, full_matrices=False)
    evr_S = (sS ** 2) / (sS ** 2).sum()
    r_s = int(torch.searchsorted(torch.cumsum(evr_S, 0), torch.tensor(s_var)).item()) + 1
    S_raw = VhS[:r_s]

    overlap = torch.linalg.svdvals(M @ S_raw.t()).max()

    S_perp = S_raw - (S_raw @ M.t()) @ M
    _, sP, VhP = torch.linalg.svd(S_perp, full_matrices=False)
    S = VhP[sP > tol].contiguous()

    diag = {"M_top1_evr": evr_M[0].item(), "r_S": S.shape[0],
            "overlap_M_S": overlap.item(),
            "S_perp_M_residual": (S @ M.t()).abs().max().item()}
    return M, S, diag


# ---------------------------------------------------------------------------------------
# plug-in module (verbatim from the advisor's reference implementation)
# ---------------------------------------------------------------------------------------
class FactorizationReg(nn.Module):
    def __init__(self, M, S, anchors_photo, anchors_sketch, *,
                 lam_leak=4.0, lam_par=6.0, warmup_steps=500,
                 normalize_par=True, l2_normalize=False):
        super().__init__()
        self.register_buffer("M", M)                          # [r_M, d]
        self.register_buffer("S", S)                          # [r_S, d]
        self.register_buffer("anchors_photo", anchors_photo)  # [C, d] frozen "a photo of a {c}"
        self.register_buffer("anchors_sketch", anchors_sketch)  # [C, d] frozen "a sketch of a {c}"
        self.lam_leak, self.lam_par = lam_leak, lam_par
        self.warmup_steps = warmup_steps
        self.normalize_par = normalize_par
        self.l2_normalize = l2_normalize

    @classmethod
    @torch.no_grad()
    def build(cls, encode_photo, encode_sketch, vocab_words, class_names, *,
              r_m=2, s_var=0.90, verbose=True, l2_normalize=False, **kw):
        """encode_photo/encode_sketch: fn(list[str]) -> Tensor [n, d] (pre-norm),
        via each branch's own frozen "a photo/sketch of a {}" template."""

        def feats(words):
            p = encode_photo(words).float()
            s = encode_sketch(words).float()
            if l2_normalize:
                p, s = F.normalize(p, dim=-1), F.normalize(s, dim=-1)
            return p, s

        vP, vS = feats(vocab_words)
        M, S, diag = estimate_subspaces(vP, vS, r_m=r_m, s_var=s_var)
        aP, aS = feats(class_names)  # per-class anchor tables
        if verbose:
            print(f"[FactorizationReg] d={M.shape[1]}  r_M={M.shape[0]}  r_S={S.shape[0]}  "
                  f"| M_top1_evr={diag['M_top1_evr']:.3f}  overlap(M,S)={diag['overlap_M_S']:.3f}")
            if diag["M_top1_evr"] < 0.5:
                print("  [warn] weak modality axis (top-PC < 0.5) -> ensemble sketch templates "
                      "or raise r_m; run the parallelism diagnostic before trusting M.")
        return cls(M, S, aP, aS, l2_normalize=l2_normalize, **kw)

    # ---- projection primitives ----
    def _sq_in(self, x, B):    # || P_B x ||^2
        return (x @ B.t()).pow(2).sum(-1)

    def _sq_out(self, x, B):   # || (I - P_B) x ||^2
        return (x.pow(2).sum(-1) - (x @ B.t()).pow(2).sum(-1)).clamp_min(0)

    def raw_losses(self, tP_tuned, tS_tuned, class_ids):
        """Unweighted L_leak, L_par for the batch (useful for logging / ablations)."""
        if self.l2_normalize:
            tP_tuned = F.normalize(tP_tuned, dim=-1)
            tS_tuned = F.normalize(tS_tuned, dim=-1)
        aP = self.anchors_photo.index_select(0, class_ids)
        aS = self.anchors_sketch.index_select(0, class_ids)

        dP = tP_tuned - aP                    # anchors are buffers -> already grad-free
        dS = tS_tuned - aS
        g = tS_tuned - tP_tuned

        l_leak = 0.5 * (self._sq_in(dP, self.S) + self._sq_in(dS, self.S)).mean()
        off = self._sq_out(g, self.M)
        if self.normalize_par:
            off = off / g.pow(2).sum(-1).clamp_min(1e-8)   # fraction of gap that's off-axis
        l_par = off.mean()
        return l_leak, l_par

    def forward(self, tP_tuned, tS_tuned, class_ids, step):
        l_leak, l_par = self.raw_losses(tP_tuned, tS_tuned, class_ids)
        ramp = min(1.0, step / max(1, self.warmup_steps))
        extra = ramp * (self.lam_leak * l_leak + self.lam_par * l_par)
        info = {"l_leak": l_leak.item(), "l_par": l_par.item(),
                "ramp": ramp, "reg_loss": extra.item()}
        return extra, info


# ---------------------------------------------------------------------------------------
# SBIR-specific wiring: frozen CLIP text encoding + ImageNet-1k vocabulary
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def _encode_frozen_text(clip_model, words, template):
    """Frozen zero-shot CLIP text feature for the literal hand-written template
    ("<template> <word>."), no learnable prompt. Only valid when the model's
    text transformer has no active deeper-layer prompt injection (text_depth=1
    -- see the assert in build_for_sbir); passing an empty deeper-prompt list
    is otherwise unsafe since injection is wired into the resblocks at
    construction time.
    """
    from src.clip import clip as _clip

    template = template.replace("_", " ")
    prompts = [f"{template} {w.replace('_', ' ')}." for w in words]
    tokenized = torch.cat([_clip.tokenize(p) for p in prompts])
    tokenized = tokenized.to(clip_model.token_embedding.weight.device)
    dtype = clip_model.dtype

    x = clip_model.token_embedding(tokenized).type(dtype)
    x = x + clip_model.positional_embedding.type(dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer([x, []])[0]
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(dtype)
    x = x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)] @ clip_model.text_projection
    return x


def _imagenet1k_classnames():
    """1000 ImageNet-1k class names, available offline via torchvision's bundled
    weights metadata (no download) -- used as a broad, seen-class-disjoint
    vocabulary for estimating M/S. See discussion: literature-standard choice
    for "broad concept coverage" in prompt-learning generalization work.
    """
    from torchvision.models import ResNet50_Weights
    return list(ResNet50_Weights.DEFAULT.meta["categories"])


def build_for_sbir(cfg, clip_photo, clip_sketch, classnames):
    """Convenience constructor: wires FactorizationReg.build() to this
    project's two CLIP branches and an ImageNet-1k vocabulary.
    """
    text_depth = getattr(cfg, 'text_depth', 1)
    assert text_depth == 1, (
        f"FactorizationReg's frozen-reference encoding is only valid for "
        f"text_depth=1 (got {text_depth}) -- deeper-layer prompt injection can't be "
        f"cleanly bypassed at runtime for text_depth > 1."
    )

    ctx_init = getattr(cfg, 'ctx_init', 'a photo of a')
    ctx_init_sketch = getattr(cfg, 'ctx_init_sketch', 'a sketch of a')

    def encode_photo(words):
        return _encode_frozen_text(clip_photo, words, ctx_init)

    def encode_sketch(words):
        return _encode_frozen_text(clip_sketch, words, ctx_init_sketch)

    vocab_words = _imagenet1k_classnames()

    return FactorizationReg.build(
        encode_photo, encode_sketch, vocab_words, classnames,
        r_m=getattr(cfg, 'leak_rank_m', 2),
        s_var=getattr(cfg, 'leak_subspace_var', 0.90),
        lam_leak=getattr(cfg, 'lambda_leak', 0.0),
        lam_par=getattr(cfg, 'lambda_par', 0.0),
        warmup_steps=getattr(cfg, 'leak_warmup_steps', 500),
        normalize_par=True,
        l2_normalize=False,  # matches this codebase's pre-norm raw text features
    )
