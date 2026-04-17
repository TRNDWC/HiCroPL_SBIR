# Fix1 vs Fix3 Ablation Guide

## 1. Goal

Run controlled experiments with exactly one deep-phase fix active at a time:
- Fix1 only
- Fix3 only

This avoids mixing behaviors and makes metric differences interpretable.

## 2. Mode Switch

Use:
- `--triplet_enable_external`
- `--triplet_mode {fix1|fix3}`

Behavior:
1. `fix1`
- Shallow: additive sketch residual on top of full text->visual flow.
- Deep: serial path (photo->sketch first, then regular visual->text on updated sketch prompts).

2. `fix3`
- Shallow: same as fix1 (kept constant for fair deep-phase comparison).
- Deep: break serial dependency and run parallel updates from original deep states:
  - Path A: photo->sketch
  - Path B: sketch->text (from original sketch state, not updated sketch)
- Apply gated residuals:
  - `visual_new = original_sketch + gate_beta * sketch_update`
  - `text_new = original_text + gate_gamma * text_update`

## 3. Why Fix3

Fix3 reduces error propagation:
- In serial deep flow, a bad photo->sketch mapping can corrupt sketch->text.
- In fix3 parallel deep flow, text branch is computed from original sketch prompts, independent of current-step sketch update.

## 4. Parameterization

Learnable gates:
- `gate_alpha` (shallow additive sketch residual)
- `gate_beta` (deep photo->sketch residual in fix3)
- `gate_gamma` (deep sketch->text residual in fix3)

All gates start near zero (`0.01`) so training starts near baseline behavior.

## 5. Prompt Parameter Counting Note

Observed issue:
- Adding sketch-photo flow seemed to increase "learnable prompt parameter" count.

Root cause:
- Previous optimizer grouping counted *all* parameters inside prompt learners as prompt params,
  including newly added mapper networks and gate scalars.

Resolution implemented:
- Prompt bucket now includes only prompt-token parameters:
  - `cross_prompts_text`
  - `cross_prompts_visual`
  - `text_proxy_tokens`
  - `visual_proxy_tokens`
- Auxiliary modules (cross-modal mappers, gates) are grouped into non-prompt params.

Result:
- Prompt parameter count now stays aligned with prompt-token design.
- Any increase from new mappers/gates appears in non-prompt count, which is expected.

## 6. Recommended Commands

Fix1 only:

```bash
python -m experiments.hicropl_prompt \
  --exp_name=triplet_fix1 \
  --triplet_enable_external \
  --triplet_mode=fix1
```

Fix3 only:

```bash
python -m experiments.hicropl_prompt \
  --exp_name=triplet_fix3 \
  --triplet_enable_external \
  --triplet_mode=fix3
```

## 7. Expected Logging

At startup (`configure_optimizers`):
- `Number of trainable prompt params: ...`
- `Number of trainable non-prompt params: ...`

Interpretation:
- Prompt count should no longer jump simply because cross-modal mapper blocks were added.
- Non-prompt count may increase with fix-specific auxiliary modules; this is correct.
