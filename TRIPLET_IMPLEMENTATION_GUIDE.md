# Triplet Cross-Modal Prompts: Implementation Guide

## 1. Final Flow Specification (as agreed)

### Early layers (0-3)
- text_sketch -> sketch (existing)
- text_photo -> photo (existing)
- sketch -> photo (new)

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
- `external_visual_to_visual_early`: for shallow `sketch -> photo`
- `external_visual_to_visual_deep`: for deep `photo -> sketch`

New forward args:
- `external_visual_prompts=None`
- `external_flow=None`

New internal method:
- `_apply_external_visual_flow(...)`

Behavior:
- `external_flow="sketch_to_photo_early"`
  - Applies only for layers `[0, cross_layer)`
  - Uses hard split-query update (2+2):
    - first query block from photo prompt attends text proxy
    - second query block from photo prompt attends sketch prompts
  - No blend coefficient is used.
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

## 3. How to Run

Enable triplet external flows during training:

```bash
python -m experiments.hicropl_prompt \
  --exp_name=hicropl_triplet \
  --triplet_enable_external
```

If you do not pass `--triplet_enable_external`, the new external flows are off and behavior stays close to previous baseline.

## 4. Mapping to Your Requirement

- Early `text -> sketch` (existing): still handled by sketch learner internal text/visual mapping.
- Early `text -> photo` (existing): still handled by photo learner internal text/visual mapping.
- Early `sketch -> photo` (new): implemented via `external_flow="sketch_to_photo_early"` in photo branch.

- Deep `photo -> text_photo` (existing): still handled by photo learner internal visual/text mapping.
- Deep `sketch -> text_sketch` (existing): still handled by sketch learner internal visual/text mapping.
- Deep `photo -> sketch` (new): implemented via `external_flow="photo_to_sketch_deep"` in sketch branch.

## 5. Practical Notes

- The implementation is minimally invasive: it keeps original HiCroPL intra-branch logic intact and only adds cross-branch injections.
- Early layer behavior is strict split-query (2+2), not weighted blending.
- Deep layer cross-modal update (`photo -> sketch`) is direct update without blend coefficient.

## 6. Files Updated

- `src/hicropl.py`
- `src/model_hicropl.py`
- `experiments/options.py`
- `TRIPLET_IMPLEMENTATION_GUIDE.md`
