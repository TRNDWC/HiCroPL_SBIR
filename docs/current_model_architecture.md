# HiCroPL-SBIR Current Model Documentation

## Status Update (2026-04-16)

This document has historical sections and is no longer fully authoritative for all bi-knowledge flows.

- Current shallow/deep cross-flow behavior should be read together with:
	- `docs/bi_knowledge_flow_additive_injection_design.md`
	- `docs/prompt_update_noise_fixes.md`
- In particular, any older statement implying strictly independent photo/sketch branches (no cross-update) should be treated as outdated.
- The source of truth is current code in:
	- `src/hicropl.py`
	- `src/model_hicropl.py`

## 1. Scope and Purpose

This document describes the current implementation of the HiCroPL-based ZS-SBIR model in this repository. The goal is to provide a detailed system-level view of:

- Model components and their roles
- How photo and sketch branches are organized
- How prompts are initialized and updated
- End-to-end forward data flow for training and evaluation
- Optimization and loss design used by the current code
- Key practical implications and constraints

The description reflects the behavior of the current source code, not an abstract paper-only design.

## 2. High-Level System Overview

The current system is organized as a four-encoder architecture plus a shared frozen distillation branch:

- Text encoder for photo branch
- Visual encoder for photo branch
- Text encoder for sketch branch
- Visual encoder for sketch branch
- Frozen CLIP reference encoder for distillation targets

Prompt learning is centralized in one shared prompt learner instance, but this prompt learner internally maintains separate prompt parameters for photo and sketch. So there is one prompt learner module with two independent parameter sets.

At training time, the model consumes:

- Sketch image
- Positive photo image
- Negative photo image
- Augmented sketch image
- Augmented photo image
- Class label

The model then produces aligned text and image features for both modalities, classification logits for both branches, and auxiliary features used by the training loss.

## 3. Module-Level Architecture

### 3.1 Core wrappers

The main runtime wrapper is the CustomCLIP module.

Responsibilities:

- Holds the shared prompt learner
- Holds branch-specific text and visual encoders
- Holds a frozen CLIP distillation branch
- Runs branch-wise forward passes and fuses prompted features with fixed distillation features

A PyTorch Lightning module wraps CustomCLIP to provide:

- Optimizer configuration
- Training step
- Validation/test logic for category-level and fine-grained settings
- Metric logging and checkpoint behavior

### 3.2 Text encoder wrapper

The text encoder wrapper injects deep text prompts into the CLIP text transformer path. It receives:

- Prepared token sequence with prompt context inserted
- Tokenized prompts
- Deeper text prompts

It returns projected text embeddings used for class-level matching.

### 3.3 Visual encoder wrapper

The visual encoder wrapper calls the CLIP visual transformer variant that supports deep visual prompt injection. It receives:

- Image tensor
- First-layer visual prompt
- Deeper visual prompts

It returns image embeddings in CLIP embedding space.

### 3.4 Prompt learner internals

The prompt learner contains two branches with explicit parameter separation:

- Photo prompt branch
- Sketch prompt branch

Each branch owns separate:

- Shallow and deep text prompts
- Shallow and deep visual prompts
- Text-to-visual mapper
- Visual-to-text mapper
- Text proxy token stack
- Visual proxy token stack
- Attention-pooling modules for proxy extraction

This explicit split allows future branch-specific logic and cross-branch visual interaction logic without parameter entanglement.

## 4. Configuration Design

The prompt learner now supports branch-specific configuration keys, with fallback to shared keys.

Branch-specific keys supported:

- prompt_depth_photo, prompt_depth_sketch
- cross_layer_photo, cross_layer_sketch
- n_ctx_photo, n_ctx_sketch
- ctx_init_photo, ctx_init_sketch
- prec_photo, prec_sketch
- dataset_photo, dataset_sketch

Fallback behavior:

- If a branch-specific key is missing, the learner uses the shared key value.
- This preserves compatibility with older configs that only define shared keys.

Compatibility aliases are also retained so older code paths that expect single-branch naming still function, defaulting to photo branch aliases.

## 5. Prompt Initialization Strategy

### 5.1 Text prompt initialization

For each branch independently:

- If branch ctx_init is provided and branch n_ctx is small enough, context tokens are initialized from token embeddings of that textual template.
- Otherwise, context tokens are initialized randomly with Gaussian initialization.

Because the two branches are initialized separately now, photo and sketch can start from different textual priors.

### 5.2 Visual prompt initialization

Visual prompts are initialized independently per branch, with branch-specific context lengths.

### 5.3 Important implication

If branch-specific initialization strings are different, the two text prompt branches start with semantically different priors from step zero.

## 6. Prompt Update Mechanics

Status note (2026-04-16): this section contains baseline-oriented wording and may not reflect later additive/cross-flow refinements.

Within each modality branch, prompt updates follow two sequential mappings:

1. Text-to-visual mapping:
- Early visual prompts are updated using text-side proxy tokens.

2. Visual-to-text mapping:
- Later text prompts are updated using visual-side proxy tokens.

Proxy extraction uses attention pooling from prompt token sequences. This creates a hierarchical bidirectional adaptation loop between text and visual prompts per branch.

No direct cross-update between photo and sketch prompt tensors is currently applied in this implementation. Branches are independent unless additional custom logic is added.

## 7. End-to-End Forward Flow During Training

### 7.1 Batch unpacking

Training forward expects batch order:

- sketch image
- positive photo image
- negative photo image
- sketch augmentation
- photo augmentation
- label

### 7.2 Prompt generation

A single call generates prompt outputs for both branches using classnames.

Each branch output includes:

- Prepared text input sequence
- Tokenized prompts
- Fixed text embeddings from frozen reference encoder
- Deeper text prompts
- Deeper visual prompts

### 7.3 Branch forward execution

Three branch passes are run:

- Photo branch for positive photo image
- Sketch branch for sketch image
- Photo branch again for negative photo image

Each branch pass computes:

- Fixed image feature from frozen reference visual encoder
- Prompted text feature from branch text encoder
- Prompted image feature from branch visual encoder
- Normalized fusion between prompted and fixed features

### 7.4 Output assembly

Forward returns a tuple containing:

- Final photo/sketch/negative image features
- Classification logits for photo and sketch branches
- Augmented image features and augmented logits
- Text features and fixed reference features for both branches

This output contract is consumed by the loss function.

## 8. Feature Fusion and Normalization Behavior

For each branch, current feature construction is:

- Prompted feature is normalized
- Fixed reference feature is normalized
- Sum is computed in embedding space
- Summed feature is normalized again

This produces a fused representation that keeps the prompted branch anchored to the frozen CLIP reference manifold.

## 9. Loss Function in Current Code

The current loss implementation computes these active terms:

- Cross-modal contrastive loss between sketch and photo features
- Cross-entropy classification loss for photo logits
- Cross-entropy classification loss for sketch logits

There are placeholders/comments for additional losses (triplet and augmented classification), but in the present active implementation the total loss is:

- contrastive cross-modal term
- plus standard classification terms

Consistency terms and some other regularizers are present as code fragments/comments but not part of the active final scalar in the current implementation.

## 10. Optimization and Trainable Parameters

### 10.1 Freezing policy

A frozen CLIP reference model is used for distillation targets.

The prompted CLIP branch is largely frozen except selected trainable components, with explicit optimizer grouping:

- Prompt learner parameters
- Non-prompt trainable parameters (mainly layer norm and any additional trainable leftovers)

### 10.2 Optimizer groups

The optimizer uses Adam with separate learning rates:

- Prompt parameter learning rate
- Non-prompt parameter learning rate

Weight decay is applied as configured.

## 11. Evaluation Pipelines

### 11.1 Category-level retrieval

Validation uses two dataloaders:

- Sketch query set
- Photo gallery set

For each sketch query:

- Similarity against gallery is computed
- Retrieval metrics include mean average precision and precision at K
- K behavior depends on dataset setting

### 11.2 Fine-grained retrieval

Features are grouped by category bucket, then per-category rank positions are computed using paired base names between sketch and photo samples.

Reported metrics include top-1, top-5, and top-10 style accuracy derived from rank statistics.

## 12. Runtime and Reproducibility Notes

The training script enforces:

- Global seed setup
- Worker-level deterministic seeding
- Deterministic trainer settings
- Fixed backbone selection in current script path

Checkpointing monitors retrieval metrics and supports resume behavior.

## 13. Current Design Strengths

- Explicit branch-level separation for prompt parameters and configs
- Shared prompt learner interface with dual branch outputs in one call
- Clean separation of prompted encoders and frozen distillation encoder
- Compatible with old config style via fallback keys
- Ready for extension with branch-specific or cross-branch interaction logic

## 14. Current Design Constraints and Risks

- There is no active explicit cross-branch prompt exchange logic yet
- Loss function currently activates fewer terms than suggested by commented structure
- Backward compatibility aliases can hide accidental branch misuse if not monitored
- Branch-specific configuration flexibility increases need for careful config validation

## 15. Suggested Next Documentation Extensions

- Add a dedicated configuration reference table with default values and valid ranges
- Add a tensor-shape flow table per major forward stage
- Add a troubleshooting section for modality mismatch and prompt depth/cross-layer constraints
- Add an ablation guide documenting expected effects of branch-specific init differences
