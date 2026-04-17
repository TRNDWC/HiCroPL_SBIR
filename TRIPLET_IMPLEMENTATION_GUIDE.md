# Triplet Cross-Modal Prompts: Implementation Guide

## 1. Final Flow Specification (as agreed)

### Early layers (0-3)
- text_sketch -> sketch (existing)
- text_photo -> photo (existing)
- sketch -> photo (new)

Shallow implementation mode (Fix 1):
- No token split.
- Keep full `text->visual` mapping on all visual tokens.
- Add sketch residual with learnable gate:
  - `updated_visual = updated_by_text + gate_alpha * sketch_contribution`
  - `gate_alpha` is initialized to `0.01`.

### Deep layers (4-8)
- photo -> text_photo (existing)
- sketch -> text_sketch (existing)
- photo -> sketch (new)

Important note:
- `text_sketch` and `text_photo` are separate by design because this codebase uses two prompt learners:
  - `prompt_learner_sketch` (text_sketch path)
  - `prompt_learner_photo` (text_photo path)

## 2. What Was Implemented

### 2.1 In src/hicropl.py
Added external cross-branch visual prompt exchange inside `CrossModalPromptLearner`.

New config fields:
- `triplet_enable_external`

New modules:
- `sketch2visual_net`: for shallow sketch residual injection
- `external_visual_to_visual_deep`: for deep `photo -> sketch`

New parameter:
- `gate_alpha = nn.Parameter(torch.tensor(0.01))`

New forward args:
- `external_visual_prompts=None`
- `external_flow=None`

New internal method:
- `_apply_external_visual_flow(...)`

Behavior:
- `external_flow="sketch_to_photo_early"`
  - Applies only for layers `[0, cross_layer)`
  - Uses additive injection over full text->visual flow:
    - `updated_by_text = text2visual_net(full_visual_tokens, text_proxy, text_proxy)`
    - `sketch_contribution = sketch2visual_net(full_visual_tokens, sketch_proxy, sketch_proxy)`
    - `updated = updated_by_text + gate_alpha * sketch_contribution`
- `external_flow="photo_to_sketch_deep"`
  - Applies only for layers `[cross_layer, prompt_depth)`
  - Injects photo visual guidance to sketch visual prompts with direct replacement (no blend).

### 2.2 In src/model_hicropl.py
Connected both new flows in `CustomCLIP`.

Changes:
- `_forward_branch(...)` accepts external flow inputs and passes them to prompt learner.
- Added `_snapshot_visual_prompts(...)` to safely pass cross-branch prompt snapshots.
- In `forward(...)`:
  - Photo branch uses `sketch_to_photo_early`.
  - Sketch branch uses `photo_to_sketch_deep`.
  - Negative photo branch uses `sketch_to_photo_early`.
- In evaluation (`extract_eval_features(...)`): same flow wiring is applied for each modality.

### 2.3 In experiments/options.py
Added CLI options:
- `--triplet_enable_external`
- `--triplet_mode` (`fix1` or `fix3`)

## 3. How to Run

Enable triplet external flows during training:

```bash
python -m experiments.hicropl_prompt \
  --exp_name=hicropl_triplet \
  --triplet_enable_external \
  --triplet_mode=fix1
```

If you do not pass `--triplet_enable_external`, the new external flows are off and behavior stays close to previous baseline.

To run Fix3 only:

```bash
python -m experiments.hicropl_prompt \
  --exp_name=hicropl_triplet_fix3 \
  --triplet_enable_external \
  --triplet_mode=fix3
```

## 4. Mapping to Your Requirement

- Early `text -> sketch` (existing): still handled by sketch learner internal text/visual mapping.
- Early `text -> photo` (existing): still handled by photo learner internal text/visual mapping.
- Early `sketch -> photo` (new): implemented via `external_flow="sketch_to_photo_early"` in photo branch.

- Deep `photo -> text_photo` (existing): still handled by photo learner internal visual/text mapping.
- Deep `sketch -> text_sketch` (existing): still handled by sketch learner internal visual/text mapping.
- Deep `photo -> sketch` (new): implemented via `external_flow="photo_to_sketch_deep"` in sketch branch.

## 5. Practical Notes

- The implementation is minimally invasive: it keeps original HiCroPL intra-branch logic intact and only adds cross-branch injections.
- Early layer behavior follows additive injection with learnable gate (`gate_alpha`).
- Deep layer cross-modal update (`photo -> sketch`) is direct update without blend coefficient.

## 6. Files Updated

- `src/hicropl.py`
- `src/model_hicropl.py`
- `experiments/options.py`
- `TRIPLET_IMPLEMENTATION_GUIDE.md`

## 7. Fix 1 Implementation Plan (Additive Injection)

This section documents the new direction:
- Shallow phase: no token splitting.
- Keep full text->visual update for all visual tokens.
- Add sketch residual with learnable gate.
- Deep phase: chain photo -> sketch -> text_sketch, no blend coefficients.

### 7.1 Required code changes

1) In src/hicropl.py
- Add parameter:
  - gate_alpha = nn.Parameter(torch.tensor(0.01))
- Add shallow mapper for sketch residual:
  - sketch2visual_net (same hidden size as visual prompt dim)
- In shallow branch (sketch_to_photo_early):
  - compute updated_by_text on full visual tokens
  - compute sketch_contribution on full visual tokens
  - update by additive gating:
    - updated = updated_by_text + gate_alpha * sketch_contribution
  - copy directly into cross_prompts_visual (no blend alpha)

2) In deep branch (photo_to_sketch_deep)
- Keep chain update style:
  - photo_proxy -> sketch update
  - sketch_proxy -> text_sketch update
- Use direct copy/update style, no blend coefficients.

3) In experiments/options.py
- No scalar blend alpha needed for this fix.
- Keep only enable flag for external flow if desired.

### 7.2 Monitoring

Track these during training:
- gate_alpha value per epoch
- retrieval metrics delta vs baseline
- gradient norms for sketch2visual_net

Interpretation:
- gate_alpha stays near zero: sketch residual not useful on current setup
- gate_alpha increases: model is exploiting sketch residual pathway

### 7.3 Why this fix is safer

- Initialization remains close to v3 behavior.
- Avoids abrupt distribution change caused by token splitting.
- Lets optimizer decide how much sketch signal to inject.
