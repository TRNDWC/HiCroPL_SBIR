"""Identity/gradient-placement check for --propagator_sketch_queried.

Runs BEFORE any real training. Builds VisualVisualPromptLearner directly (no
image data needed -- Z_l/Pp_l are the raw [n_ctx, dim] prompt tensors, no
batch dimension), forwards once, and:

  1. Prints cos(P_hat_l, Z_l) per layer, immediately after init.
  2. If --propagator_gate_mode zero_init: asserts torch.allclose(P_hat, Q)
     EXACTLY for every layer. Exits non-zero on failure -- do not train on a
     zero_init run that fails this.
  3. Runs one fake backward (loss = P_hat.sum()) and asserts:
     a. current_photo_prompts[0:cross_layer] receive exactly zero gradient
        (confirms photo stays read-only).
     b. every self.q_proj[l].weight has a non-None, nonzero-norm gradient.
     c. at least one parameter inside self.pool[l] has gradient, for every l.

Usage:
    python scripts/check_propagator_identity.py --propagator_gate_mode none
    python scripts/check_propagator_identity.py --propagator_gate_mode zero_init
    python scripts/check_propagator_identity.py --propagator_gate_mode learned
    python scripts/check_propagator_identity.py --propagator_gate_mode zero_init \
        --propagator_bottleneck_rank 32
    python scripts/check_propagator_identity.py --n_ctx 3 --prompt_depth 12 --cross_layer 12
"""
import argparse
import sys
import types

import torch

sys.path.insert(0, "/Users/tranduc/nckh/ZS-SBIR/Sketch_VLM")

from src.hicropl import VisualVisualPromptLearner


def build_cfg(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--n_ctx', type=int, default=3)
    p.add_argument('--prompt_depth', type=int, default=12)
    p.add_argument('--cross_layer', type=int, default=12)
    p.add_argument('--propagator_gate_mode', type=str, default='none',
                    choices=['none', 'zero_init', 'learned'])
    p.add_argument('--propagator_bottleneck_rank', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)

    cfg = types.SimpleNamespace(
        n_ctx=args.n_ctx, prompt_depth=args.prompt_depth, cross_layer=args.cross_layer,
        disable_exchange=False, exchange_detach_source=False, exchange_detach_lkp_input=False,
        exchange_sketch_driven=False, exchange_query_from_sketch=False,
        exchange_bottleneck_rank=0, exchange_sketch_retrieval=False,
        exchange_free_source=False, exchange_self_source=False, exchange_no_mapper=False,
        exchange_no_lkp=False, mapper_single_scale=False, sketch_self_refine=False,
        sketch_self_refine_ln=False, proxy_init='randn', n_proxy=1,
        propagator_sketch_queried=True,
        propagator_gate_mode=args.propagator_gate_mode,
        propagator_bottleneck_rank=args.propagator_bottleneck_rank,
    )
    return cfg, args


def make_fake_clip(p_dim=768, dtype=torch.float32):
    """Minimal stand-in exposing only what VisualVisualPromptLearner.__init__ reads."""
    clip = types.SimpleNamespace()
    clip.dtype = dtype
    clip.visual = types.SimpleNamespace()
    clip.visual.conv1 = types.SimpleNamespace(weight=torch.zeros(p_dim, 3, 32, 32, dtype=dtype))
    return clip


def main():
    cfg, args = build_cfg(sys.argv[1:])
    torch.manual_seed(args.seed)

    p_dim = 768
    clip = make_fake_clip(p_dim)
    vl = VisualVisualPromptLearner(cfg, clip, clip)  # no sample images -> plain Gaussian init
    assert vl.cross_layer == vl.prompt_depth, (
        "this script's P_hat reconstruction assumes cross_layer == prompt_depth (n_deep=0), "
        "matching every --propagator_sketch_queried run this task defines; pass matching values."
    )

    print(f"=== check_propagator_identity.py | gate_mode={cfg.propagator_gate_mode} | "
          f"bottleneck_rank={cfg.propagator_bottleneck_rank} | "
          f"n_ctx={cfg.n_ctx} cross_layer={cfg.cross_layer} prompt_depth={cfg.prompt_depth} ===\n")

    # ---- forward, keep the pre-update Q for comparison ----
    Q_before = torch.cat([vl.cross_prompts_sketch[l] for l in range(vl.cross_layer)], dim=0).detach().clone()
    photo_shallow, sketch_shallow, photo_deeper, sketch_deeper = vl()
    P_hat = torch.cat([sketch_shallow.unsqueeze(0)] + [d.unsqueeze(0) for d in sketch_deeper[:vl.cross_layer - 1]],
                       dim=0).view(-1, p_dim)
    # sketch_shallow is layer 0, sketch_deeper[0..] are layers 1.. -- P_hat above
    # reassembles exactly the [cross_layer, n_ctx, dim] block the propagator wrote.

    n_ctx = vl.n_ctx
    P_hat_by_layer = P_hat.view(vl.cross_layer, n_ctx, p_dim)
    Q_by_layer = Q_before.view(vl.cross_layer, n_ctx, p_dim)

    print("cos(P_hat_l, Z_l) per layer:")
    all_cos_one = True
    for l in range(vl.cross_layer):
        cos = torch.nn.functional.cosine_similarity(
            P_hat_by_layer[l].flatten(), Q_by_layer[l].flatten(), dim=0).item()
        flag = "" if abs(cos - 1.0) < 1e-6 else "  <-- not 1.0"
        print(f"  layer {l:2d}: cos = {cos:+.8f}{flag}")
        if abs(cos - 1.0) >= 1e-6:
            all_cos_one = False

    if cfg.propagator_gate_mode == 'zero_init':
        ok = torch.allclose(P_hat, Q_before, atol=0.0, rtol=0.0)
        print(f"\n[zero_init] torch.allclose(P_hat, Q) EXACT: {'PASS' if ok else 'FAIL'}")
        if not ok:
            diff = (P_hat - Q_before).abs()
            print(f"[zero_init] FAIL DETAIL: max abs diff = {diff.max().item():.6e}, "
                  f"mean abs diff = {diff.mean().item():.6e}")
            print("[zero_init] REFUSING to proceed -- fix mapper_update's zero-init before training.")
            sys.exit(1)
    else:
        print(f"\n[{cfg.propagator_gate_mode}] cos==1.0 for all layers (no gate expected): "
              f"{'yes' if all_cos_one else 'no (expected -- no identity guarantee for this mode)'}")

    # ---- fake backward ----
    vl.zero_grad()
    loss = P_hat.sum()
    loss.backward()

    print("\n=== Gradient placement checks ===")
    fail = []

    # (a) photo prompts must receive EXACTLY zero (or no) gradient
    photo_ok = True
    for l in range(vl.cross_layer):
        g = vl.cross_prompts_photo[l].grad
        if g is not None and not torch.allclose(g, torch.zeros_like(g)):
            photo_ok = False
            fail.append(f"cross_prompts_photo[{l}] got nonzero grad (norm={g.norm().item():.6f})")
    print(f"(a) cross_prompts_photo[0..{vl.cross_layer-1}] grad == 0 (read-only): "
          f"{'PASS' if photo_ok else 'FAIL'}")

    # (b) q_proj must all have gradient
    qproj_ok = True
    for l in range(vl.cross_layer):
        g = vl.q_proj[l].weight.grad
        if g is None or g.norm().item() == 0:
            qproj_ok = False
            fail.append(f"q_proj[{l}].weight has no/zero grad")
    print(f"(b) q_proj[0..{vl.cross_layer-1}].weight grad != None and norm > 0: "
          f"{'PASS' if qproj_ok else 'FAIL'}")

    # (c) each pool[l] has at least one param with grad
    pool_ok = True
    for l in range(vl.cross_layer):
        has = any(p.grad is not None and p.grad.norm().item() > 0 for p in vl.pool[l].parameters())
        if not has:
            pool_ok = False
            fail.append(f"pool[{l}] has no parameter with nonzero grad")
    print(f"(c) pool[0..{vl.cross_layer-1}] has >=1 param with grad != None, norm > 0: "
          f"{'PASS' if pool_ok else 'FAIL'}")

    # (b)/(c) at the VERY FIRST backward are only a meaningful pass/fail
    # criterion for gate_mode='none': there, nothing blocks the gradient path,
    # so q_proj/pool SHOULD have gradient immediately.
    #
    # For 'zero_init'/'learned', a zero gradient at q_proj/pool on step 0 is
    # mathematically FORCED, not a bug: zero_init zeroes attn.out_proj and
    # ffn.c_proj, and a zero weight MATRIX blocks gradient in both forward and
    # backward identically -- only out_proj/c_proj themselves (the params
    # multiplying the zero) receive gradient; everything upstream (linear_q/k/
    # v, attn.in_proj, ffn.c_fc, and hence q_proj/pool) gets exactly zero.
    # 'learned' has the same symptom via a different mechanism: gate=tanh(0)=0
    # scales the WHOLE delta path to zero gradient, only gamma escapes (same
    # ReZero bootstrap already documented for --exchange_sketch_driven).
    #
    # The property that actually matters for these two modes is not "grad
    # reaches q_proj/pool on step 0" (provably false) but "grad reaches
    # q_proj/pool once the gate has moved". Verified below by manually nudging
    # the gate (matches how --exchange_sketch_driven's bootstrap was verified
    # earlier in this project) and redoing forward+backward.
    if cfg.propagator_gate_mode in ('zero_init', 'learned'):
        print(f"\n=== [{cfg.propagator_gate_mode}] step-0 zero grad on q_proj/pool is EXPECTED "
              f"(ReZero bootstrap) -- verifying gradient escapes once the gate opens ===")
        with torch.no_grad():
            if cfg.propagator_gate_mode == 'zero_init':
                vl.mapper_update.attn.out_proj.weight.add_(0.01)
                vl.mapper_update.ffn.c_proj.weight.add_(0.01)
            else:
                vl.gamma.add_(0.05)
        vl.zero_grad()
        _, sketch_shallow2, _, sketch_deeper2 = vl()
        P_hat2 = torch.cat([sketch_shallow2] + list(sketch_deeper2[:vl.cross_layer - 1]), dim=0)
        P_hat2.sum().backward()
        qproj_ok2 = all(vl.q_proj[l].weight.grad is not None and vl.q_proj[l].weight.grad.norm().item() > 0
                        for l in range(vl.cross_layer))
        pool_ok2 = all(any(p.grad is not None and p.grad.norm().item() > 0 for p in vl.pool[l].parameters())
                       for l in range(vl.cross_layer))
        print(f"    after nudging the gate away from 0: q_proj grad>0 for all layers = "
              f"{'PASS' if qproj_ok2 else 'FAIL'}, pool grad>0 for all layers = "
              f"{'PASS' if pool_ok2 else 'FAIL'}")
        qproj_ok, pool_ok = qproj_ok2, pool_ok2  # this is the criterion that actually gates PASS/FAIL below
        fail = [m for m in fail if 'q_proj' not in m and 'pool[' not in m]
        if not qproj_ok2:
            fail.append("q_proj still has zero grad even after the gate was nudged open -- real bug")
        if not pool_ok2:
            fail.append("pool still has zero grad even after the gate was nudged open -- real bug")

    all_ok = photo_ok and qproj_ok and pool_ok and (
        cfg.propagator_gate_mode != 'zero_init' or torch.allclose(P_hat, Q_before, atol=0.0, rtol=0.0))

    print(f"\n>>> OVERALL: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        for msg in fail:
            print(f"    - {msg}")
        sys.exit(1)


if __name__ == '__main__':
    main()
