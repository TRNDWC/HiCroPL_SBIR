# Bi-Knowledge Flow Fix 3 Design: Parallel Deep-Phase Updates

## 1. Goal

This document describes a design-only change for deep-phase bi-knowledge flow to avoid serial dependency between:

- photo -> sketch update
- sketch -> text update

The core idea is to break the serial chain and compute both paths from the same original deep state.

## 2. Problem Statement

In a serial deep pipeline, if sketch prompts are first updated by photo and then reused to update text, errors from photo->sketch can propagate into the text branch.

This creates two risks:

- unstable optimization when cross-modal mapping quality is poor early in training
- coupled failure mode where one branch corrupts the other

Fix 3 addresses this by making deep updates parallel and state-isolated.

## 3. Key Principle

At each deep forward pass:

1. Capture original deep sketch prompts and original deep text prompts.
2. Build two independent update paths from originals.
3. Apply gated residual updates independently.

No path should consume the intermediate output of the other path in the same pass.

## 4. Parallel Deep Paths

### Path A: photo -> sketch

- Input query: original deep sketch visual prompts.
- Input key/value: photo deep proxies (from external deep photo information or pooled photo deep prompts).
- Output: sketch update residual.

### Path B: sketch -> text

- Input query: original deep text prompts.
- Input key/value: sketch proxies computed from original deep sketch prompts.
- Output: text update residual.

Important: Path B must not use sketch prompts that have already been updated by Path A.

## 5. Update Rule

For each deep layer index i:

- sketch_new[i] = original_sketch[i] + gate_beta * sketch_update[i]
- text_new[i] = original_text[i] + gate_gamma * text_update[i]

Then write both updates back.

This makes both branches residual, controllable, and decoupled.

## 6. Gating Strategy

Use separate learnable gates:

- gate_beta for photo->sketch deep residual
- gate_gamma for sketch->text deep residual

Recommended initialization:

- small positive values (for example 0.01)

Rationale:

- preserves baseline-like behavior at initialization
- lets model gradually increase cross-branch influence if useful

Optional constraint:

- keep gates non-negative via parameterization (softplus) if needed

## 7. State Isolation Requirements

Before any deep update is applied, snapshot:

- original deep sketch visual prompts
- original deep text prompts

All deep pooling and attention inputs must be built from these snapshots.

Do not read in-place updated prompts while computing the second path.

## 8. Shape and Compatibility Notes

The design assumes deep tensors can be aligned per layer between:

- sketch deep visual prompts
- text deep prompts
- photo deep proxies

If deep lengths differ between branches:

- define explicit alignment policy (min length, truncate, or repeat)
- keep policy deterministic and documented

## 9. Expected Benefits

- reduced error propagation across branches
- better stability in early and mid training
- cleaner ablation analysis because paths are explicitly independent
- safer integration with additional cross-modal flows

## 10. Potential Trade-offs

- slightly higher memory due to snapshots
- extra compute for dual-path deep attention
- gate tuning may be required for best gains

## 11. Suggested Monitoring

Track during training:

- gate_beta and gate_gamma trajectories
- norm of sketch_update and text_update residuals
- deep branch loss stability indicators
- retrieval metrics (mAP, P@K, top-k)

## 12. Ablation Plan

Minimum experiments:

1. current deep serial behavior
2. parallel deep update without gates (fixed 1.0)
3. parallel deep update with learnable gates
4. parallel deep update with small fixed gates

Compare:

- convergence stability
- retrieval metrics
- sensitivity to initialization

## 13. Scope

This is a design document only.

- No code changes are included here.
- No API change is required by this design alone.
- Intended as pre-implementation reference for the deep-phase refactor.
