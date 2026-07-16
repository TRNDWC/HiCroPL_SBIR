"""
Modality-semantic factorization regularizer for the prompt-tuned text branch.

Decomposes each class's prompted text-feature nudge (relative to the frozen,
un-prompted CLIP embedding of its own template) into a modality component M
(the photo<->sketch direction) and its orthogonal complement S (the semantic/
category directions):

    L_leak = || P_S d ||^2        protect class-semantic subspace S (both branches)
    L_par  = || (I - P_M) g ||^2  keep the sketch/photo gap on the shared axis M
                                   (normalized by ||g||^2 -> scale-free in [0,1])

Adapted from a reference plug-in; wired to this repo's CLIP tokenizer/encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.clip import clip as _clip


def make_encode_text_fn(clip_model):
    """Wrap a frozen CLIP model's encode_text to accept raw strings."""
    device = next(clip_model.parameters()).device

    @torch.no_grad()
    def encode_text(strings):
        tokens = _clip.tokenize(strings).to(device)
        return clip_model.encode_text(tokens)

    return encode_text


# ---------------------------------------------------------------------------------------
# core subspace estimation
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
    threshold = torch.tensor(s_var, device=evr_S.device, dtype=evr_S.dtype)
    r_s = int(torch.searchsorted(torch.cumsum(evr_S, 0), threshold).item()) + 1
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
# plug-in module
# ---------------------------------------------------------------------------------------
class FactorizationReg(nn.Module):
    def __init__(self, M, S, anchors_photo, anchors_sketch, *,
                 lam_leak=4.0, lam_par=6.0, warmup_steps=500,
                 normalize_par=True, l2_normalize=False):
        super().__init__()
        self.register_buffer("M", M)                          # [r_M, d]
        self.register_buffer("S", S)                          # [r_S, d]
        self.register_buffer("anchors_photo", anchors_photo)   # [C, d] frozen "a photo of a {c}"
        self.register_buffer("anchors_sketch", anchors_sketch) # [C, d] frozen "a sketch of a {c}"
        self.lam_leak, self.lam_par = lam_leak, lam_par
        self.warmup_steps = warmup_steps
        self.normalize_par = normalize_par
        self.l2_normalize = l2_normalize

    # ---- convenience constructor: does everything from a frozen encoder ----
    @classmethod
    @torch.no_grad()
    def build(cls, encode_text, vocab_words, class_names, *,
              template_photo="a photo of a {}", template_sketch="a sketch of a {}",
              r_m=2, s_var=0.90, verbose=True, **kw):
        def feats(words):
            p = encode_text([template_photo.format(w) for w in words]).float()
            s = encode_text([template_sketch.format(w) for w in words]).float()
            if kw.get("l2_normalize", False):
                p, s = F.normalize(p, dim=-1), F.normalize(s, dim=-1)
            return p, s

        vP, vS = feats(vocab_words)
        M, S, diag = estimate_subspaces(vP, vS, r_m=r_m, s_var=s_var)
        aP, aS = feats(class_names)                            # per-class anchor tables
        if verbose:
            print(f"[FactorizationReg] d={M.shape[1]}  r_M={M.shape[0]}  r_S={S.shape[0]}  "
                  f"| M_top1_evr={diag['M_top1_evr']:.3f}  overlap(M,S)={diag['overlap_M_S']:.3f}")
            if diag["M_top1_evr"] < 0.5:
                print("  [warn] weak modality axis (top-PC < 0.5) -> ensemble sketch templates "
                      "or raise r_M; run the parallelism diagnostic before trusting M.")
        return cls(M, S, aP, aS, **kw)

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
