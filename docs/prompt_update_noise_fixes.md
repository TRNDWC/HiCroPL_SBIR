# Prompt Update Noise Fixes

## Scope

This note documents fixes applied to reduce unintended prompt-state drift and serial update noise.

## Fixed Issues

1. Validation prompt drift across batches
- Previous behavior: each call to `extract_eval_features` triggered `prompt_learner.forward_all(...)`, which mutates prompt tensors in place.
- Effect: validation features in the same epoch were computed from evolving prompt states.
- Fix: cache `prompt_outputs` once per validation epoch and reuse for all validation batches.

2. Serial shallow dependency in `forward_all`
- Previous behavior: photo shallow external sketch proxies were read after sketch branch had already updated in the same `forward_all` pass.
- Effect: photo update consumed in-pass mutated sketch state, adding ordering sensitivity and extra noise.
- Fix: snapshot sketch shallow visual prompts before branch updates, then build photo external proxies from this snapshot.

## Code Changes

- `src/model_hicropl.py`
  - Added `self._cached_eval_prompt_outputs` in `HiCroPL_SBIR`.
  - Added `on_validation_epoch_start` to precompute and cache prompt outputs once.
  - Updated `extract_eval_features` to reuse cache and lazily initialize if needed.
  - Cleared cache in `on_validation_epoch_end`.

- `src/hicropl.py`
  - Extended `_build_external_shallow_visual_proxies_from_sketch(...)` to accept an optional sketch snapshot source.
  - Extended `_forward_single_modality(...)` to accept `sketch_shallow_snapshot`.
  - Updated `forward_all(...)` to snapshot sketch shallow prompts before updates and pass snapshot to photo branch.

## Expected Outcome

- More stable validation metrics within the same epoch.
- Reduced order-dependent noise in photo shallow prompt updates.
- Cleaner A/B comparison between logic variants, with less hidden prompt-state drift.

## Current Policy

- Training still mutates prompt state once per batch by design.
- Validation and test reuse one cached prompt snapshot per epoch to avoid repeated mutation noise across batches.
