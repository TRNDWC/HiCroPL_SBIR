# Sketch_VLM: HiCroPL-SBIR Architecture Documentation

**Project**: CLIP for All Things Zero-Shot Sketch-Based Image Retrieval  
**Paper**: CVPR 2023 - "CLIP for All Things Zero-Shot Sketch-Based Image Retrieval, Fine-Grained or Not"  
**Time**: 2024-2025

---

## 1. Project Overview

### 1.1 Problem Statement
**Zero-Shot Sketch-Based Image Retrieval (ZS-SBIR)** aims to retrieve photo gallery images given a sketch query without seeing any photo-sketch pairs during training. The model must generalize to unseen object categories and handle both:
- **Category-level retrieval**: Finding photos of the same category as the sketch
- **Fine-grained retrieval**: Finding the exact same object instance

### 1.2 Core Innovation: HiCroPL (Hierarchical Cross-modal Prompt Learning)

The architecture leverages **CLIP** (Contrastive Language-Image Pre-training) with a novel prompt learning approach:

1. **Sketch-specific prompts**: Learn task-specific prompts rather than relying on generic CLIP
2. **Cross-modal prompts**: Enable bidirectional knowledge flow between text and visual domains
3. **Deep prompt injection**: Inject learnable prompts at multiple transformer layers
4. **Hierarchical design**: Use Layer-Specific Knowledge Proxies (LKP) to capture multi-scale features

### 1.3 Key Features
- ✅ Zero-shot generalization to unseen categories
- ✅ Instance-level fine-grained matching through patch shuffling
- ✅ Dual-branch architecture (photo & sketch modalities)
- ✅ Knowledge distillation from frozen CLIP teacher
- ✅ Consistency regularization through augmented views

---

## 2. Architecture Overview

### 2.1 High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    HiCroPL-SBIR System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐         ┌──────────────────┐            │
│  │  CustomCLIP      │         │  HiCroPL_SBIR    │            │
│  │ (Wrapper Layer)  │────────▶│ (Lightning Module)            │
│  └──────────────────┘         └──────────────────┘            │
│         │                          │         │                │
│         ├─ Photo Branch            ├─ Training                 │
│         ├─ Sketch Branch           ├─ Validation               │
│         └─ Frozen Distill          └─ Test                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 CustomCLIP: Four-Branch Architecture

```
                 Input Batch
                     │
         ┌───────────┼───────────┐
         │           │           │
         ▼           ▼           ▼
      Sketch      Photo      Negative
        ⌊──────┬───⌋          ⌊──┬──⌋
        │      │              │  │  
        │    ┌─▼─────────────────▼──┐         ┌──────────────┐
        │    │ Photo Prompt Learner │◀────────│  Distill     │
        │    └──────────┬──────────┘          │  Teacher CLIP│
        │               │                      └──────────────┘
        │    ┌──────────▼──────────┐
        ├───▶│ Sketch Prompt       │
        │    │  Learner            │
        │    └──────────┬──────────┘
        │               │
        └───────────────┼─────────────────────┐
                        │                     │
          ┌─────────────▼────────┐    ┌──────▼──────┐
          │ Text Encoder (Photo) │    │ Visual      │
          │ Text Encoder         │    │ Encoder     │
          │ (Sketch)             │    │ (Photo/Sketch)
          └─────────────┬────────┘    └──────┬──────┘
                        │                     │
                        └─────────────┬───────┘
                                      │
                        ┌─────────────▼────────┐
                        │ Feature Fusion &     │
                        │ Logit Computation    │
                        └─────────────┬────────┘
                                      │
                                  Output
```

---

## 3. Core Architecture Components

### 3.1 CrossModalPromptLearner

**Purpose**: Learn bidirectional prompts for both text and visual domains

**Key Parameters**:
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_ctx` | 4 | Number of learnable context tokens |
| `prompt_depth` | 9 | Number of transformer layers to inject prompts |
| `cross_layer` | 4 | Layer index where cross-modal flow starts |
| `ctx_init` | "a photo of a" | Initial prompt text initialization |

**Architecture**:

```
                 Classnames
                     │
        ┌────────────▼────────────┐
        │ Tokenize & Embed        │
        │ (Fixed CLIP token_emb)  │
        └────────────┬────────────┘
                     │
           ┌─────────▼──────────┐
           │ Construct Prompts  │
           │ [prefix|ctx|suffix]│
           └─────────┬──────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        │ Layer 0-3              │ Layer 4-8
        │ (Early Cross-Modal)    │ (Late Visual)
        │                         │
        ├─ T→I Mapping           ├─ I→T Mapping
        │  (Text guides visual)  │  (Visual guides text)
        │                         │
        └────────────┬────────────┘
                     │
          ┌──────────▼──────────┐
          │ Updated Prompts     │
          │ cross_prompts_text  │
          │ cross_prompts_visual│
          └──────────┬──────────┘
                     │
            [text_input, tokenized_prompts,
             fixed_embeddings, cross_prompts_*]
```

**Three Key Objects**:

1. **cross_prompts_text** (`ParameterList` of 9 tensors)
   - Shape: `[n_ctx=4, 512]` for each layer
   - `512`: Text embedding dimension in CLIP
   - Updated via I→T mapping (visual knowledge influences text)

2. **cross_prompts_visual** (`ParameterList` of 9 tensors)
   - Shape: `[n_ctx=4, 768]` for each layer
   - `768`: Visual embedding dimension (ViT-B/32 patch size)
   - Updated via T→I mapping (text knowledge influences visual)

3. **Layer-Specific Knowledge Proxies (LKP)**
   - AttentionPooling modules extract aggregated layer information
   - Used as keys/values in cross-modal attention mechanisms

### 3.2 Knowledge Mapper Networks

#### 3.2.1 T→I Mapper (`text2visual_net`)
Maps text prompt information into visual prompt space:

```
Input: visual_prompts (shallow layers)
       proxy_text_prompts (text knowledge proxies)

Process:
  Q: visual_prompts → linear_q
  K: proxy_text → linear_k (project to visual space)
  V: proxy_text → linear_v
  
  multi_head_attention(Q, K, V) + FFN + LayerNorm

Output: Updated visual_prompts
        (initialized with aligned text knowledge)
```

#### 3.2.2 I→T Mapper (`visual2text_net`)
Maps visual information back into text space:

```
Input: text_prompts (deeper layers)
       proxy_visual_prompts (visual knowledge proxies)

Process:
  Q: text_prompts → linear_q
  K: proxy_visual → linear_k
  V: proxy_visual → linear_v
  
  multi_head_attention(Q, K, V) + FFN + LayerNorm

Output: Updated text_prompts
```

### 3.3 Attention Pooling Layers

Each AttentionPooling layer:
- **Input**: Query token + sequence (key/value)
- **Operation**: Self-attention over sequence, aggregated into query
- **Output**: Single aggregated token representing layer's knowledge

Used to extract proxy tokens representing the "essence" of each prompt layer.

### 3.4 TextEncoder & VisualEncoder

#### TextEncoder
```python
def forward(prompts, tokenized_prompts, cross_prompts_text_deeper):
    x = prompts + positional_embedding           # Add position info
    x = x.permute(1, 0, 2)                       # NLD → LND
    
    # Pass through transformer with side injection
    combined = [x, cross_prompts_text_deeper]    # Multi-branch input
    x = transformer(combined)                     # Frozen CLIP transformer
    
    # Extract text features at EOS token
    x = ln_final(x)
    x = x[batch_idx, eot_idx] @ text_projection   # Project to 512-dim space
    
    return x  # [batch_size, 512]
```

**Key Points**:
- Leverages frozen CLIP text transformer
- Injects deep prompts at transformer input
- Extracts features at End-of-Sequence (EOS) token (standard CLIP behavior)

#### VisualEncoder
```python
def forward(image, first_visual_prompt, deeper_visual_prompts):
    # Delegates to modified ViT backbone
    return vit(image, first_visual_prompt, deeper_visual_prompts)
    # Returns [batch_size, 768]
```

Wraps ViT with deep prompt injection capability.

---

## 4. Training Pipeline

### 4.1 Data Flow in CustomCLIP.forward()

```
Input: batch = [sketch, photo, negative, sk_aug, photo_aug, label]
       classnames = list of category names

Step 1: Generate Prompts
  ├─ prompt_learner_photo(classnames)
  │  └─ Returns: text_features, image_features_fixed, prompts
  └─ prompt_learner_sketch(classnames)
     └─ Returns: text_features, image_features_fixed, prompts

Step 2: Extract Features (4 branches)
  ├─ Photo branch:    photo_feat, text_feat_photo
  ├─ Sketch branch:   sketch_feat, text_feat_sketch
  ├─ Negative branch: neg_feat (from photo encoder)
  └─ Augmented:       photo_aug_feat, sketch_aug_feat

Step 3: Feature Fusion
  ├─ image_features_norm1 = image_features / ||image_features||
  ├─ image_features_prenorm = image_features_norm1 + image_features_fixed
  ├─ image_features_final = image_features_prenorm / ||image_features_prenorm||
  │  (Residual connection + L2 normalization)
  └─ Same for text features

Step 4: Compute Logits
  ├─ logits_photo = scale * photo_feat @ text_feat_photo.T
  └─ logits_sketch = scale * sketch_feat @ text_feat_sketch.T

Output: 14-tuple for loss computation
  (photo_feat, logits_photo,
   sketch_feat, logits_sketch,
   neg_feat, label,
   photo_aug_feat, sketch_aug_feat,
   logits_photo_aug, logits_sketch_aug,
   text_feat_photo, text_feat_sketch,
   text_feat_fixed_photo, text_feat_fixed_sketch)
```

### 4.2 Feature Construction Flow

```
Raw Image Features → Raw + Fixed Fusion → Normalization

photo_feat = encoder(photo)   [B, 512]
           ↓
           + Fixed reference (from frozen CLIP)
           ↓
           Residual connection + E.Norm
           ↓
           Final normalized feature [B, 512]

Key Insight: 
- Fixed reference acts as stability anchor
- E.Norm ensures features lie on unit hypersphere
- Residual connection enables smooth blending
```

---

## 5. Loss Function: loss_fn_hicropl

### 5.1 Loss Components

The total loss combines multiple objectives:

```
Total Loss = L_cross_modal + L_ce
           = λ_cm × L_InfoNCE(sketch, photo)
           + λ_ce × [L_CE(photo_logits, label) + L_CE(sketch_logits, label)]
```

### 5.2 Loss Details

#### L1: Cross-Modal InfoNCE Loss (REMOVED in current version)
```
Purpose: Align sketch and positive photo in embedding space
Form: Contrastive loss with in-batch negatives
Status: Commented out (triplet loss variant)
```

#### L2: Cross-Modal Alignment (Active)
```python
def cross_loss(feature_1, feature_2, temperature=0.07):
    """
    InfoNCE loss between two batches
    
    Process:
    1. Normalize features to unit norm
    2. Concatenate: features = [feature_1, feature_2]  # 2B samples
    3. Compute all-to-all similarity: S = features @ features.T
    4. Create positive pairs: positive[i] = similarity[i, i+B] for i<B
    5. Negative samples: all other entries
    6. Temperature scaling + cross-entropy
    """
    labels = torch.cat([torch.arange(len(feature_1)) for _ in range(2)])
    similarities_matrix = features @ features.T
    
    # Remove diagonal (self-similarity)
    mask = torch.eye(...)
    positives = similarities_matrix[labels.bool()]      # 1 positive per sample
    negatives = similarities_matrix[~labels.bool()]      # 2B-2 negatives per sample
    
    logits = torch.cat([positives, negatives], dim=1) / temperature
    return cross_entropy(logits, zero_labels)
```

**Hyperparameters**:
- `temperature`: 0.07 (default CLIP value)
- `λ_cross_modal`: 1.0

#### L3: Consistency Regularization (DISABLED)
```python
# Originally intended to enforce consistency between:
# - sketch and sketch_aug
# - photo and photo_aug
# Status: Commented out to simplify loss
```

#### L4: Classification Loss (Active)
```python
loss_ce = λ_ce × [cross_entropy(logits_photo, label) 
                   + cross_entropy(logits_sketch, label)]
```

**Why**:
- Ensures learned features align with CLIP's text classification
- Provides dense supervision signal
- Acts as self-supervised task

**Hyperparameters**:
- `λ_ce`: 1.0

### 5.3 Optimizer Configuration

```python
def configure_optimizers():
    # Group 1: Prompt parameters (learnable)
    prompt_params = [
        model.prompt_learner_photo.parameters(),
        model.prompt_learner_sketch.parameters()
    ]
    
    # Group 2: LayerNorm parameters (frozen backbone)
    ln_params = [ln.parameters() for ln in model.modules() if isinstance(ln, LayerNorm)]
    
    # Group 3: Other trainable parameters
    extra_params = [all other params with requires_grad=True]
    
    # Different learning rates
    optimizer = Adam([
        {'params': prompt_params, 'lr': prompt_lr=1e-5},
        {'params': ln_params + extra_params, 'lr': clip_ln_lr=1e-5}
    ], weight_decay=1e-4)
```

**Key Design**:
- Separate learning rates for different parameter groups
- Lower learning rates to preserve pretrained CLIP quality
- Weight decay regularization

---

## 6. Backbone Freezing Strategy

### 6.1 Model Freezing Hierarchy

```
CustomCLIP Structure:
├─ clip_model_distill (Frozen CLIP)
│  └─ visual_encoder: freeze_model() ✓ ALL PARAMS FROZEN
│
├─ clip_photo (Deep-copy of base CLIP)
│  ├─ apply(freeze_all_but_bn)
│  │  └─ Only LayerNorm: trainable ✓
│  │  └─ All others: frozen ✓
│  └─ Wrapped by: prompt_learner_photo, text_encoder_photo, visual_encoder_photo
│
└─ clip_sketch (Deep-copy of base CLIP)
   ├─ apply(freeze_all_but_bn)
   │  └─ Only LayerNorm: trainable ✓
   │  └─ All others: frozen ✓
   └─ Wrapped by: prompt_learner_sketch, text_encoder_sketch, visual_encoder_sketch
```

### 6.2 Trainable Parameters

Only these are trained:

1. **Prompt Learner Parameters** (HiCroPL-specific):
   - `cross_prompts_text[0..8]`
   - `cross_prompts_visual[0..8]`
   - Attention pooling modules
   - Knowledge mapper networks

2. **LayerNorm Bias/Weight**:
   - All `nn.LayerNorm` modules throughout CLIP

3. **Logit Scale**: 
   - Temperature parameter for similarity scaling

### 6.3 Why This Strategy?

```
Goal: Efficient fine-tuning while preserving CLIP knowledge

Benefits:
- Dramatically reduces memory (prompt << backbone)
- Faster training (fewer parameters to optimize)
- Better generalization (maintains pretrained quality)
- Stable optimization (not changing entire backbone)

Risk: Might limit expressiveness
Solution: HiCroPL's deep prompt injection compensates
```

---

## 7. Validation & Evaluation Flow

### 7.1 Category-Level Evaluation

```python
def _validation_step_category(batch, dataloader_idx):
    """Process each validation batch"""
    
    if dataloader_idx == 0:  # Sketch dataset
        sketch_feat = extract_eval_features(tensor, 'sketch')
        collect(sketch_feat)
    else:  # dataloader_idx == 1, Photo dataset
        photo_feat = extract_eval_features(tensor, 'photo')
        collect(photo_feat)

def _on_validation_epoch_end_category():
    """Compute mAP and Precision@k metrics"""
    
    # Collect all features
    gallery_features = cat(photo_features)      # [N_photo, 512]
    query_features = cat(sketch_features)       # [N_sketch, 512]
    
    # Compute similarity matrix
    similarity_matrix = query_features @ gallery_features.T  # [N_sketch, N_photo]
    
    # For each sketch query:
    for i in range(N_sketch):
        category_i = sketch_labels[i]
        distances_i = similarity_matrix[i]      # [N_photo]
        target = (photo_labels == category_i)   # Ground truth
        
        # Compute metrics
        mAP_i = retrieval_average_precision(distances_i, target)
        precision_i = retrieval_precision(distances_i, target, k=top_k)
    
    # Average across all queries
    final_mAP = mean(mAP_i for all i)
    final_precision = mean(precision_i for all i)
```

**Metrics**:
- **mAP (mean Average Precision)**: Standard IR metric
- **P@k (Precision@k)**: Fraction of top-k results that match category
  - P@100 for Sketchy
  - P@200 for Sketchy-Extended

### 7.2 Fine-Grained Evaluation

```python
def _validation_step_fg(batch, dataloader_idx):
    """Collect fine-grained features per category"""
    
    # Extract feature for each image
    feature = extract_eval_features(tensor)
    
    # Group by category
    fg_buckets[category_idx]['features'].append(feature)
    fg_buckets[category_idx]['filenames'].append(filename)
    fg_buckets[category_idx]['base_names'].append(base_name)

def _on_validation_epoch_end_fine_grained():
    """Compute rank-based accuracy for fine-grained matching"""
    
    all_ranks = []
    for category in categories:
        # Within-category retrieval
        sketch_feats = stack(fg_sketches[category])
        photo_feats = stack(fg_photos[category])
        
        # Compute distances
        sim_matrix = sketch_feats @ photo_feats.T
        
        # For each sketch, find rank of its matching photo
        for sketch_idx in range(len(sketch_feats)):
            sketch_base_name = sketch_base_names[sketch_idx]
            
            # Find matching photo
            matching_photo_idx = photo_base_names.index(sketch_base_name)
            
            # Get rank: how many photos rank higher?
            distances = sim_matrix[sketch_idx]
            rank = (distances > distances[matching_photo_idx]).sum() + 1
            
            all_ranks.append(rank)
    
    # Compute accuracy
    acc@1 = mean(rank == 1 for rank in all_ranks)
    acc@5 = mean(rank <= 5 for rank in all_ranks)
    acc@10 = mean(rank <= 10 for rank in all_ranks)
```

**Key Difference from Category-Level**:
- Matches based on instance identity (same object)
- Requires parsing filenames to extract base object identity
- More challenging task

---

## 8. Data Pipeline

### 8.1 Dataset Structure

```
data_dir/
├── sketch/
│   ├── apple/
│   │   ├── apple_1.png
│   │   ├── apple_2.png
│   │   └── ...
│   ├── bicycle/
│   │   └── ...
│   └── ...
│
└── photo/
    ├── apple/
    │   ├── apple_photo_1.jpg
    │   ├── apple_photo_2.jpg
    │   └── ...
    ├── bicycle/
    │   └── ...
    └── ...
```

### 8.2 Unseen Classes

For Zero-Shot evaluation, specific categories are held out:

```python
UNSEEN_CLASSES = {
    "sketchy": [
        "bat", "cabin", "cow", "dolphin", "door", "giraffe", ...
    ],  # 21 categories for zero-shot evaluation
    ...
}
```

**Split**:
- **Train**: All categories EXCEPT unseen_classes
- **Val/Test**: ONLY unseen_classes

This ensures true zero-shot evaluation.

### 8.3 Sketchy Dataset (__getitem__)

```python
def __getitem__(index):
    # Get sketch path from index
    sketch_path = all_sketches_path[index]
    category = sketch_path.parent.name
    
    # Sample positive photo (same category)
    positive_photo = random_choice(photos_in_category[category])
    
    # Sample negative photo (different category)
    neg_category = random_choice(other_categories)
    negative_photo = random_choice(photos_in_category[neg_category])
    
    # Load and resize images
    sketch = ImageOps.pad(Image.open(sketch_path), (224, 224))
    photo = ImageOps.pad(Image.open(positive_photo), (224, 224))
    negative = ImageOps.pad(Image.open(negative_photo), (224, 224))
    
    # Apply augmentation
    sk_tensor = transform(sketch)
    img_tensor = transform(photo)
    neg_tensor = transform(negative)
    
    sk_aug_tensor = augmentation(sketch)       # Strong augmentation
    img_aug_tensor = augmentation(photo)
    
    return (sk_tensor, img_tensor, neg_tensor, sk_aug_tensor, img_aug_tensor, 
            category_index, filename)
```

**Data Augmentation**:

```python
# Normal transform (evaluation)
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Strong augmentation (training)
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
    transforms.Normalize(...)
])
```

---

## 9. Training Procedure

### 9.1 Initialization

```python
# 1. Load pretrained CLIP models
clip_model = load_clip_to_cpu(opts)           # For training
clip_model_frozen = load_clip_to_cpu_teacher(opts)  # Teacher (frozen)

# 2. Create CustomCLIP wrapper
model = CustomCLIP(opts, clip_model, clip_model_frozen)

# 3. Create Lightning module
plmodel = HiCroPL_SBIR(opts, classnames, model)

# 4. Create datasets
train_dataset = Sketchy(opts, transform, mode='train')
val_sketch = ValidDataset(opts, mode='sketch')
val_photo = ValidDataset(opts, mode='photo')

# 5. Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=192, shuffle=True)
val_sketch_loader = DataLoader(val_sketch, shuffle=False)
val_photo_loader = DataLoader(val_photo, shuffle=False)
```

### 9.2 Training Loop (PyTorch Lightning)

```python
for epoch in range(max_epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        # 1. Forward pass
        features = model(batch, classnames)
        
        # 2. Compute loss
        loss = loss_fn_hicropl(args, features)
        
        # 3. Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        # Run on validation sets
        for batch in val_sketch_loader:
            sketch_feat = model.extract_eval_features(batch, 'sketch')
            collect_features(sketch_feat)
        
        for batch in val_photo_loader:
            photo_feat = model.extract_eval_features(batch, 'photo')
            collect_features(photo_feat)
        
        # Compute metrics
        mAP, Precision = compute_metrics()
        
        # Save checkpoint if best
        if mAP > best_metric:
            save_checkpoint()
```

### 9.3 on_train_epoch_start

```python
def on_train_epoch_start(self):
    # Set encoders to eval mode (important!)
    self.model.visual_encoder_photo.eval()
    self.model.visual_encoder_sketch.eval()
    self.model.text_encoder_photo.eval()
    self.model.text_encoder_sketch.eval()
    self.model.clip_model_distill.eval()
```

**Why?**:
- These encoders contain frozen backbone + learnable prompts
- Setting eval() mode disables dropout and batch norm statistics updates
- Only trainable prompts affect gradients

---

## 10. Key Implementation Details

### 10.1 Feature Normalization Strategy

```python
# Raw features from encoder
image_features = encoder(image)  # [B, D]

# Step 1: L2 normalization
image_features_norm1 = image_features / image_features.norm(dim=-1, keepdim=True)

# Step 2: Add fixed reference (residual)
image_features_prenorm = image_features_norm1 + image_features_fixed

# Step 3: Final normalization
image_features_final = image_features_prenorm / image_features_prenorm.norm(dim=-1, keepdim=True)

Technical insight:
- Fixed reference acts as curriculum signal
- Residual connection preserves both dynamic and static components
- Final normalization ensures features are on unit sphere
- This improves stability and prevents feature collapse
```

### 10.2 Distillation Branch

```python
self.clip_model_distill = clip_model_frozen  # Frozen CLIP
self.distill_visual_encoder = self.clip_model_distill.visual

# During training:
with torch.no_grad():
    photo_aug_feat = self.distill_visual_encoder(photo_aug)  # Reference features
    sketch_aug_feat = self.distill_visual_encoder(sketch_aug)

# These features are used for:
# - Consistency loss targets
# - Fixed reference embeddings for fusion
```

### 10.3 Prompt Caching

```python
class CrossModalPromptLearner:
    def __init__(self):
        self._cached_classnames = None
        self._cached_text_input = None
        self._cached_tokenized_prompts = None
        self._cached_fixed_embeddings = None
    
    def _prepare_dynamic_classnames(self, classnames):
        # Check if already cached
        if self._cached_classnames == classnames:
            return cached_values
        
        # Otherwise compute and cache
        text_input = self.construct_prompts(...)
        tokenized = clip.tokenize(...)
        fixed = self.clip_model_distill.encode_text(...)
        
        self._cached_classnames = classnames
        # ... cache others ...
        
        return text_input, tokenized, fixed
```

**Benefit**: Avoids recomputing prompts multiple times per batch

### 10.4 Batch Processing

Each training batch structure:
```
Input: 
  [sketch_batch, photo_batch, neg_batch, 
   sketch_aug_batch, photo_aug_batch,
   label_batch, filename_batch]

Processing:
  - Sketch, Photo, Negative → 3 forward passes through 4-branch model
  - Extract image features + text features
  - Fuse with fixed references
  - Compute logits
  - Aggregate outputs for loss
```

---

## 11. Hyperparameter Configuration

### 11.1 Model Configuration (options.py)

```python
# CLIP backbone
backbone = 'ViT-B/32'           # Vision Transformer

# Prompt Learning
n_ctx = 4                        # Context length
prompt_depth = 9                 # 9 transformer layers
cross_layer = 4                  # Layer split point
ctx_init = "a photo of a"        # Initial text

# Training
batch_size = 192
epochs = 50
prompt_lr = 1e-5                 # Prompt learning rate
clip_ln_lr = 1e-5                # LayerNorm learning rate
weight_decay = 1e-4

# Loss
temperature = 0.07
lambda_cross_modal = 1.0
lambda_ce = 1.0
lambda_consistency = 1.0
triplet_margin = 0.3

# Data
max_size = 224
workers = 8  # DataLoader workers
```

### 11.2 Dataset Configuration

```python
dataset = 'sketchy'              # Sketchy / QuickDraw / TUBerlin
data_dir = '/path/to/data'
data_split = 0.0                 # 0 = full split, >0 = random fraction

# Evaluation
eval_mode = 'category'           # or 'fine_grained'
test_batch_size = 64
```

---

## 12. Design Rationale & Key Insights

### 12.1 Why Hierarchical Cross-Modal Prompts?

```
Problem: Standard CLIP is optimized for photo-text matching, not sketch-photo

Solution hierarchy:
1. Sketch-specific prompts adapt CLIP to sketch domain
   └─ More effective than generic prompts

2. Cross-modal prompts enable information fusion
   └─ Text guides visual features (T→I)
   └─ Visual guides text features (I→T)

3. Hierarchical injection at multiple layers
   └─ Shallow layers: category-level semantics
   └─ Deep layers: instance-level details
```

### 12.2 Why Deep Prompt Injection?

```
Motivation: Different transformer layers capture different information

Layer 0-3:
  - Early cross-modal fusion enabled (LKP with attention pooling)
  - Text proxies guide visual prompt initialization
  - Semantic alignment

Layer 4-8:
  - Late visual dominance (less text interaction)
  - Fine details encoded
  - Instance distinction
```

### 12.3 Residual Feature Fusion

```
Why add fixed reference?
  
  fixed_feat = frozen_encoder(image)  # Golden reference
  dynamic_feat = trainable_encoder(image)
  
  fused = (dynamic_feat + fixed_feat) / norm

Benefits:
- Stability: fixed acts as regularizer
- Knowledge preservation: both streams needed
- Smooth optimization: easier gradient flow
```

### 12.4 Why Freeze CLIP Backbone?

```
Trade-off analysis:

Full Fine-tuning:
  ✓ Maximum expressiveness
  ✗ Huge parameter count
  ✗ Risk of overfitting
  ✗ May break zero-shot generalization

Prompt Learning (current):
  ✓ Efficient (only 0.1% parameters)
  ✓ Preserves pretrained knowledge
  ✓ Better generalization to unseen classes
  ✓ Stable optimization
  ✗ Limited expressiveness (but compensated by deep injection)
```

---

## 13. Failure Modes & Debugging

### 13.1 Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Feature collapse | All features identical | Check normalization, reduce lr |
| Training diverges | Loss → NaN | Reduce learning rate, check temperature |
| Bad validation | Metrics don't improve | Check data split (unseen classes) |
| OOM | GPU memory error | Reduce batch_size or prompt_depth |
| LayerNorm params not trained | Metrics plateau | Verify `freeze_all_but_bn` logic |

### 13.2 Debugging Checklist

```python
# 1. Feature statistics
print(f"Feature norm: {feat.norm(dim=-1).mean()}")  # Should be ~1.0
print(f"Feature range: [{feat.min()}, {feat.max()}]")

# 2. Gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm()}")

# 3. Loss components
print(f"L_cross_modal: {loss_cross_modal.item()}")
print(f"L_ce: {loss_ce.item()}")

# 4. Epoch validation
if not model.test_photo_features:
    print("WARNING: No validation features collected!")

# 5. Category consistency
print(f"Train categories: {set(train_categories)}")
print(f"Val categories: {set(val_categories)}")
assert train_categories.isdisjoint(val_categories), "Data leak!"
```

---

## 14. Evaluation Metrics Explained

### 14.1 Mean Average Precision (mAP)

```
For each query sketch i:
  1. Rank all gallery photos by similarity
  2. For top-k results, compute precision at each recall level
  3. Average these precisions (AP_i)

mAP = mean(AP_i for all queries)

Range: [0, 1]
Better: Higher is better

mAP@k: Only consider top-k results (e.g., mAP@200)
```

### 14.2 Precision@k

```
P@k = (# of correct results in top-k) / k

Example:
  - Top-5 results: [✓ cat, ✓ cat, ✗ dog, ✓ cat, ✗ bird]
  - P@5 = 3/5 = 0.6

Range: [0, 1]
```

### 14.3 Fine-Grained Accuracy@k

```
Acc@k = (# sketches where matching photo is in top-k) / (total sketches)

Example:
  - Query sketch: "cat_1.png"
  - Matching photo: "cat_photo_1.jpg"
  - If top-1 result is matching photo: Acc@1 +1
  
Range: [0, 1]
Top-1 accuracy is the hardest (must be exact match)
```

---

## 15. System Execution Flow (End-to-End)

```
┌───────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                 │
├───────────────────────────────────────────────────┤
│ • Load CLIP backbone (fp32)                       │
│ • Freeze all except LayerNorm                     │
│ • Initialize HiCroPL prompt learners              │
│ • Create deep prompt injection networks           │
└──────────┬────────────────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────┐
│ 2. DATA LOADING                                   │
├───────────────────────────────────────────────────┤
│ • Load categories (separate train/val/test)       │
│ • Create triplets: (sketch, pos_photo, neg_photo)│
│ • Apply augmentations                             │
│ • Create batches [B, 7] each                      │
└──────────┬────────────────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────┐
│ 3. TRAINING LOOP (per epoch)                      │
├───────────────────────────────────────────────────┤
│ For each batch:                                   │
│   a) Generate prompts via CrossModalPromptLearner │
│   b) Forward through 4-branch network             │
│   c) Extract and fuse features                    │
│   d) Compute loss components                      │
│   e) Backward pass                                │
│   f) Update prompts and LayerNorms                │
│                                                   │
│ Loss: InfoNCE (sketch-photo) + Classification    │
└──────────┬────────────────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────┐
│ 4. VALIDATION (per epoch)                         │
├───────────────────────────────────────────────────┤
│ Sketch dataset batch:                             │
│   → extract_eval_features(sketch, 'sketch')       │
│   → collect features                              │
│                                                   │
│ Photo dataset batch:                              │
│   → extract_eval_features(photo, 'photo')         │
│   → collect features                              │
│                                                   │
│ Compute metrics:                                  │
│   • Similarity matrix: query_feat @ gallery_feat  │
│   • For each query: compute mAP, precision        │
│   • Log metrics                                   │
└──────────┬────────────────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────┐
│ 5. CHECKPOINTING                                  │
├───────────────────────────────────────────────────┤
│ • Save model if mAP improves                      │
│ • Track best metrics                              │
│ • Continue training or stop                       │
└───────────────────────────────────────────────────┘
```

---

## 16. Configuration & Execution

### 16.1 Training Command

```bash
python -m experiments.hicropl_prompt \
    --exp_name=hicropl_sbir \
    --n_prompts=4 \
    --prompt_depth=9 \
    --cross_layer=4 \
    --clip_LN_lr=1e-5 \
    --prompt_lr=1e-5 \
    --batch_size=192 \
    --workers=8 \
    --epochs=50 \
    --dataset=sketchy
```

### 16.2 Evaluation Command

```bash
python -m experiments.hicropl_prompt \
    --eval_mode=category \
    --eval_dataset=sketchy \
    --resume_from=saved_models/hicropl_sbir/best.ckpt
```

---

## 17. File Organization

```
src/
├── model_hicropl.py          # CustomCLIP + HiCroPL_SBIR (main model)
├── hicropl.py                # CrossModalPromptLearner components
├── losses_hicropl.py         # loss_fn_hicropl definition
├── dataset_retrieval.py       # Sketchy dataset
├── utils.py                  # CLIP loading utilities
└── clip/                      # CLIP backbone
    ├── clip.py
    ├── model.py
    └── simple_tokenizer.py

experiments/
├── hicropl_prompt.py         # Main training script
└── options.py                # Configuration

src_fg/
├── utils_fg.py               # Fine-grained utilities
└── ...
```

---

## 18. Future Improvements & Research Directions

### 18.1 Short-term Enhancements
- [ ] Enable consistency loss (L3) for better regularization
- [ ] Add triplet loss (L1) back with better margin tuning
- [ ] Experiment with different prompt initialization strategies
- [ ] Implement multi-modal augmentation (e.g., sketch-like photo transforms)

### 18.2 Long-term Research
- [ ] Investigate other vision backbones (ViT-L/14, etc.)
- [ ] Extend to open-vocabulary retrieval
- [ ] Adaptive prompt depth per task
- [ ] Hierarchical category-aware prompts
- [ ] Cross-modal metric learning improvements

---

## References

1. **Paper**: "CLIP for All Things Zero-Shot Sketch-Based Image Retrieval, Fine-Grained or Not" - CVPR 2023
2. **HiCroPL Original**: Hierarchical Cross-Modal Prompt Learning for Zero-Shot SBIR
3. **CLIP**: Learning Transferable Models for Computer Vision Tasks
4. **Project**: https://github.com/aneeshan95/Sketch_LVM

---

**Documentation Version**: 1.0  
**Last Updated**: April 17, 2026  
**Author**: Architecture Analysis Team

