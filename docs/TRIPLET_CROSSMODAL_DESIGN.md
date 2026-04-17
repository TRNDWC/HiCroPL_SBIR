# Triplet Cross-Modal Prompt Learning: Design Document

**Concept**: Extend HiCroPL with direct Sketch ↔ Photo cross-modal prompts in addition to existing Text ↔ Photo/Sketch flows.

**Version**: 1.0  
**Date**: April 17, 2026

---

## 1. Core Motivation

### 1.1 Current Architecture Limitation

Current HiCroPL only has:
```
Text ←→ Visual (Photo)
  ↑
  └─── Text ←→ Visual (Sketch)
```

**Problem**: 
- Sketch and Photo only meet through Text intermediary
- No direct structural alignment between modalities
- Misses direct cross-domain feature correspondences
- Fine-grained matching less effective (instance-level details lost)

### 1.2 Proposed Enhancement: Triplet Cross-Modal Flow

Add explicit **Sketch ↔ Photo** knowledge transfer:

```
Text ←────────────┐
  ↑               │
  │               ↓
  ├──→ Photo ←────→ Sketch
  │               ↑
  └───────────────┘

New connections:
  1. Photo → Sketch (direct)
  2. Sketch → Photo (direct)
```

### 1.3 Intuition

**Why Sketch ↔ Photo matters**:
- **Sketches capture structure**: Edges, contours, spatial layouts
- **Photos capture semantics**: Texture, color, fine details
- **ZS-SBIR task**: Match structural patterns (no training data for these categories)

By enabling direct coupling:
- Photo encoder learns to emphasize structural/edge features
- Sketch encoder learns semantic grounding
- Both improve retrieval accuracy

---

## 2. Detailed Architecture: Triplet Flow

### 2.1 Layer Split & Prompt Organization

```
Layer 0-3: Early Triplet Coupling (Parallel influence on Photo)
├── Text ──→ Sketch (existing)
├── Text ──→ Photo (existing T→I)
├── Sketch ──→ Photo (NEW S→P) ⭐ [PARALLEL to Text influence]
└── Photo ↔ Sketch (existing bidirectional)

Layer 4-8: Deep Multi-Source Architecture (Parallel multi-input flows)
├── Photo ──→ Text_photo (existing P→T)
├── Photo ──→ Sketch (NEW P→S direct) ⭐
├── Sketch ──→ Text_sketch (existing S→T)
└── (Sketch & Photo independently feed their respective text paths)
```

**Key Difference**:
- **Early (0-3)**: Photo receives BOTH Text AND Sketch influence **in parallel**
  - Text guides semantic understanding of photo
  - Sketch guides structural understanding of photo
  
- **Deep (4-8)**: Text receives BOTH Photo AND Sketch inputs via separate paths
  - Photo directly updates text_photo branch
  - Sketch directly updates text_sketch branch (existing path)
  - Photo also directly influences sketch (structural feedback)

---

### 2.1.1 What Happens at Each Layer Type?

**Early Layers (0-3) - PARALLEL INPUTS TO PHOTO**:
```
Photo Prompt receives BOTH sources in parallel:

Source 1: Text influence
  Text proxy ──→ Cross-Attn ──→ photo_attn_text [n_ctx/2, 768]

Source 2: Sketch influence (NEW)
  Sketch proxy ──→ Cross-Attn ──→ photo_attn_sketch [n_ctx/2, 768]

Merge:
  merged = concat(photo_attn_text, photo_attn_sketch)  # [n_ctx, 768]
  photo_updated = merged

Result: Photo learns BOTH text semantics AND sketch structure!
```

**Deep Layers (4-8) - PARALLEL OUTPUTS FROM PHOTO & SKETCH**:
```
Photo Prompts (unchanged topology, but enhanced influence)
  ├─ Path 1: Photo → Text_photo (existing)
  │           └─→ Updates text features with visual semantics
  │
  └─ Path 2: Photo → Sketch (NEW) ⭐
             └─→ Direct visual structure guidance for sketch

Sketch Prompts (enhanced with new incoming edge)
  ├─ Receives: Photo → Sketch (NEW)
  │            └─→ Refines sketch with photo's structural details
  │
  └─ Path: Sketch → Text_sketch (existing)
           └─→ Filtered abstraction reaches text layer

Result: Text gets dual input (photo_direct + sketch_filtered)
        Sketch enriched with photo's concrete features
```

---

### 2.1.2 Why Different Architectures?

| Aspect | Early (Parallel to Photo) | Deep (Parallel to Text) |
|--------|------------------------|----------------------|
| **Goal** | Balanced multi-source influence on visual encoder | Dual semantic pathways to text encoder |
| **Photo role** | Receives from text & sketch | Provides direct guidance to sketch & text |
| **Sketch role** | Influences photo + receives from text | Receives from photo, filters to text |
| **Text role** | Bidirectional partner (influences photo) | Receives from photo AND sketch branches |
| **Flow** | Diverge (Text/Sketch) → Merge (Photo) | Parallel (Photo→T & S→T) with cross-link (P→S) |
| **Why?** | Photo is hub: gets diverse inputs early | Text is hub: gets filtered + direct inputs deep |

### 2.2 Early Layers (0-3): Split Photo Prompt

**New concept**: Partition photo prompt into **two sub-streams**

```
Original photo_prompt[i] (shape: [n_ctx=4, 768])
  │
  ├─ Split ────→ photo_prompt_text[i] (shape: [n_ctx=2, 768])
  │              # Will query against text knowledge
  │
  └─ Split ────→ photo_prompt_sketch[i] (shape: [n_ctx=2, 768])
                 # Will query against sketch knowledge

Processing:

Step 1: Generate Knowledge Proxies
  text_proxy = AttnPool(text_prompts[i]) → shape [1, 512]
  sketch_proxy = AttnPool(sketch_prompts[i]) → shape [1, 768]

Step 2: Cross-Attention (Parallel)
  
  Path A (Text influence):
    Q: photo_prompt_text[i] (shape: [n_ctx/2, 768])
    K/V: text_proxy (shape: [1, 512]) → project to [1, 768]
    
    → Project K/V to photo's embedding space:
       K_proj = linear_k(text_proxy)  # [1, 768]
       V_proj = linear_v(text_proxy)  # [1, 768]
    
    → multi_head_attn(Q, K_proj, V_proj)
    → photo_attn_text (shape: [n_ctx/2, 768])
  
  Step B (Sketch influence - NEW):
    Q: photo_prompt_sketch[i] (shape: [n_ctx/2, 768])
    K/V: sketch_proxy (already [1, 768])
    
    → multi_head_attn(Q, sketch_proxy, sketch_proxy)
    → photo_attn_sketch (shape: [n_ctx/2, 768])

Step 3: Merge Two Streams
  updated_photo_prompt[i] = concat([photo_attn_text, photo_attn_sketch])
                          # Shape: [n_ctx=4, 768] ✓ Same as original
  
  # With residual connection:
  updated_photo_prompt[i] = photo_prompt[i] + 0.5 * updated_photo_prompt[i]
                            (balanced blend to avoid training instability)
```

**Benefit of splitting**:
- Explicit representation: photo learns from TWO sources simultaneously
- Separate gradient paths for text vs sketch influence
- No information mixing before cross-attention
- Easier to analyze which modality dominates at test time

### 2.3 Deep Layers (4-8): Multi-Source Feature Fusion

**Concept**: Separate but parallel influence paths - Photo and Sketch independently update Text and each other

```
Layer 4-8 Processing Order:

Path A: Photo → Sketch (NEW direct visual guidance)
─────────────────────────────────────────────
Photo prompts[i] 
  ↓
  Extract photo_proxy = AttnPool(photo_prompts[i])
  ↓
  Cross-Attention(Q=sketch_prompts[i], K/V=photo_proxy)
  ↓
Updated Sketch prompts[i]  ⭐ (refined with photo's structure)


Path B: Photo → Text_photo (EXISTING)
─────────────────────────────────
Photo prompts[i]
  ↓
  Extract photo_proxy (as above)
  ↓
  Cross-Attention(Q=text_prompts[i], K/V=photo_proxy)
  ↓
Updated Text_photo prompts[i]  ✅ (unchanged, visual semantics)


Path C: Sketch → Text_sketch (NEW semantic filtering)
──────────────────────────────────────────────────
Sketch prompts[i] (already updated from Path A)
  ↓
  Extract sketch_proxy = AttnPool(updated_sketch_prompts[i])
  ↓
  Cross-Attention(Q=text_prompts[i], K/V=sketch_proxy)
  ↓
Updated Text_sketch prompts[i]  ⭐ (abstract structure to text)
```

**Why Parallel instead of Chain?**

Parallel (Proposed):
```
Photo ──→ Text_photo  (direct visual semantics)
  │
  └──→ Sketch ──→ Text_sketch  (filtered structural abstraction)

Benefit: Text gets BOTH direct (concrete) and filtered (abstract) signals
         No single bottleneck
         Sketch enriched early (doesn't wait for text update)
         More robust: if one path weak, other compensates
```

vs Chain (Photo → Sketch → Text):
```
Photo ──→ Sketch ──→ Text  (single linear path)

Limitation: Text only sees sketch's abstraction, loses photo details
            Sequential dependency (slower optimization)
            Sketch must process before text update
```

**Advantage of Parallel**:
- Photo's structural details reach Sketch immediately (Path A)
- Text gets quick access to Photo features (Path B) 
- Text also gets filtered Sketch perspective (Path C)
- Better information diversity for text features

---

## 3. Implementation Details

### 3.1 New CrossModalPromptLearner Structure

```python
class CrossModalPromptLearner_Triplet(nn.Module):
    def __init__(self, cfg, clip_model, clip_model_distill=None):
        super().__init__()
        
        # Existing components
        self.prompt_depth = cfg.prompt_depth  # 9
        self.cross_layer = cfg.cross_layer    # 4
        self.n_ctx = cfg.n_ctx                # 4
        
        # ========== NEW: Triplet configuration ==========
        self.triplet_enabled = getattr(cfg, 'triplet_enabled', True)
        self.early_triplet_layers = cfg.cross_layer  # Layers 0-3
        self.deep_chain_layers = cfg.prompt_depth - cfg.cross_layer  # Layers 4-8
        
        # Existing prompts
        self.cross_prompts_text = ...   # [9, n_ctx, 512]
        self.cross_prompts_visual = ... # [9, n_ctx, 768]
        
        # ========== NEW: Split photo prompts for early layers ==========
        # For layers 0 to cross_layer-1: split into text and sketch components
        self.photo_prompt_text_components = nn.ParameterList([
            nn.Parameter(torch.empty(n_ctx // 2, 768, dtype=dtype))
            for _ in range(self.early_triplet_layers)
        ])
        self.photo_prompt_sketch_components = nn.ParameterList([
            nn.Parameter(torch.empty(n_ctx // 2, 768, dtype=dtype))
            for _ in range(self.early_triplet_layers)
        ])
        # Initialize with small random values
        for p in list(self.photo_prompt_text_components) + list(self.photo_prompt_sketch_components):
            nn.init.normal_(p, std=0.02)
        
        # ========== NEW: Cross-attention networks for S↔P ==========
        # Early layers: Photo ↔ Sketch attention
        self.sketch_to_photo_net = nn.ModuleList([
            CrossPromptAttention(hidden_size=768, encoder_hidden_size=768, num_attention_heads=8)
            for _ in range(self.early_triplet_layers)
        ])
        
        self.photo_to_sketch_net = nn.ModuleList([
            CrossPromptAttention(hidden_size=768, encoder_hidden_size=768, num_attention_heads=8)
            for _ in range(self.early_triplet_layers)
        ])
        
        # Deep layers: Chain (Photo → Sketch → Text)
        self.photo_to_sketch_chain = nn.ModuleList([
            CrossPromptAttention(hidden_size=768, encoder_hidden_size=768, num_attention_heads=8)
            for _ in range(self.deep_chain_layers)
        ])
        
        self.sketch_to_text_chain = nn.ModuleList([
            CrossPromptAttention(hidden_size=512, encoder_hidden_size=768, num_attention_heads=8)
            for _ in range(self.deep_chain_layers)
        ])
        
        # ========== NEW: Proxy extraction for sketch ↔ photo ==========
        self.sketch_proxy_nets = nn.ModuleList([
            AttentionPooling(hidden_size=768, num_attention_heads=8)
            for _ in range(self.early_triplet_layers)
        ])
        
        self.photo_proxy_nets_early = nn.ModuleList([
            AttentionPooling(hidden_size=768, num_attention_heads=8)
            for _ in range(self.early_triplet_layers)
        ])
        
        self.photo_proxy_nets_deep = nn.ModuleList([
            AttentionPooling(hidden_size=768, num_attention_heads=8)
            for _ in range(self.deep_chain_layers)
        ])
        
        self.sketch_proxy_nets_deep = nn.ModuleList([
            AttentionPooling(hidden_size=768, num_attention_heads=8)
            for _ in range(self.deep_chain_layers)
        ])
```

### 3.2 Pseudocode: New Forward Pass

```python
def forward(self, classnames):
    # Prepare text prompts (unchanged)
    text_input, tokenized_prompts, fixed_embeddings = self._prepare_dynamic_classnames(classnames)
    
    # ========== EARLY LAYERS (0 to cross_layer-1): Triplet coupling ==========
    for layer_idx in range(self.cross_layer):
        # Generate proxies from current prompts
        text_proxy = self.attn_pooling_text_nets[layer_idx](
            token_query=self.text_proxy_tokens[layer_idx],
            sequence_key=self.cross_prompts_text[layer_idx],
            sequence_value=self.cross_prompts_text[layer_idx]
        )  # [1, 512]
        
        sketch_proxy = self.sketch_proxy_nets[layer_idx](
            token_query=...,  # NEW
            sequence_key=self.cross_prompts_visual[layer_idx],
            sequence_value=self.cross_prompts_visual[layer_idx]
        )  # [1, 768]
        
        # NEW: Split photo prompt processing
        if layer_idx < self.early_triplet_layers:
            # Path A: Text influence on photo_text_component
            photo_text_component_updated = self.sketch_to_photo_net[layer_idx](
                q=self.photo_prompt_text_components[layer_idx],  # [n_ctx/2, 768]
                k=text_proxy[:, :],  # Project from [1, 512] to [1, 768]
                v=text_proxy[:, :]
            )  # [n_ctx/2, 768]
            
            # Path B: Sketch influence on photo_sketch_component (NEW)
            photo_sketch_component_updated = self.photo_to_sketch_net[layer_idx](
                q=self.photo_prompt_sketch_components[layer_idx],  # [n_ctx/2, 768]
                k=sketch_proxy,  # [1, 768]
                v=sketch_proxy
            )  # [n_ctx/2, 768]
            
            # Merge: update visual prompts
            merged_photo_prompt = torch.cat([
                photo_text_component_updated,
                photo_sketch_component_updated
            ], dim=0)  # [n_ctx, 768]
            
            # Residual blend
            self.cross_prompts_visual[layer_idx].data = (
                self.cross_prompts_visual[layer_idx] + 0.5 * merged_photo_prompt
            )
        
        # Also update sketch prompts (existing logic)
        updated_sketch = self.visual2text_net(...)  # As before
        self.cross_prompts_visual[layer_idx].data.copy_(updated_sketch)
    
    # ========== DEEP LAYERS (cross_layer to prompt_depth): Multi-source parallel updates ==========
    for layer_idx in range(self.cross_layer, self.prompt_depth):
        chain_idx = layer_idx - self.cross_layer
        
        # Extract photo proxy (used in BOTH paths)
        photo_proxy_deep = self.photo_proxy_nets_deep[chain_idx](
            token_query=...,
            sequence_key=self.cross_prompts_visual[layer_idx],
            sequence_value=self.cross_prompts_visual[layer_idx]
        )  # [1, 768]
        
        # ===== PATH A: Photo → Sketch (NEW direct visual guidance) =====
        updated_sketch_from_photo = self.photo_to_sketch_chain[chain_idx](
            q=self.cross_prompts_visual[layer_idx],
            k=photo_proxy_deep,
            v=photo_proxy_deep
        )  # [n_ctx, 768]
        
        # Update sketch prompts (no blend)
        self.cross_prompts_visual[layer_idx].data.copy_(updated_sketch_from_photo)
        
        # ===== PATH B: Photo → Text_photo (EXISTING, unchanged) =====
        updated_text_photo_from_photo = self.text_projection_net[chain_idx](
            q=self.cross_prompts_text[layer_idx],
            k=photo_proxy_deep,
            v=photo_proxy_deep
        )  # [n_ctx, 512]
        
        # (Could update text directly, or keep existing - depends on implementation)
        # Option: self.cross_prompts_text[layer_idx] updates as before
        
        # ===== PATH C: Sketch → Text_sketch (NEW filtered semantic flow) =====
        sketch_proxy_deep = self.sketch_proxy_nets_deep[chain_idx](
            token_query=...,
            sequence_key=self.cross_prompts_visual[layer_idx],  # Now updated from photo
            sequence_value=self.cross_prompts_visual[layer_idx]
        )  # [1, 768]
        
        updated_text_sketch_from_sketch = self.sketch_to_text_chain[chain_idx](
            q=self.cross_prompts_text[layer_idx],
            k=sketch_proxy_deep,  # Project from [1, 768] to text space
            v=sketch_proxy_deep
        )  # [n_ctx, 512]
        
        # Update text prompts as in current modal update style (no blend coefficient)
        self.cross_prompts_text[layer_idx].data.copy_(updated_text_sketch_from_sketch)
        
        # Option B: Apply sequentially
        # self.cross_prompts_text[layer_idx].data = updated_text_photo_from_photo
        # self.cross_prompts_text[layer_idx].data += 0.3 * updated_text_sketch_from_sketch
    
    # Return updated prompts and embeddings
    return text_input, tokenized_prompts, fixed_embeddings, cross_prompts_text_deeper, cross_prompts_visual_deeper
```

---

## 4. Data Flow Visualization

### 4.1 Early Layers (0-3): Triplet Update Loop

```
                    ┌─────────────────┐
                    │  Cross-layer=4  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
    Layer 0            Layer 1  ...          Layer 3
        │                    │                    │
        │  ┌────────────┐    │  ┌────────────┐    │
        │  │Extract     │    │  │Extract     │    │
        │  │Proxies:    │    │  │Proxies:    │    │
        │  │- text      │    │  │- text      │    │
        │  │- sketch    │    │  │- photo     │    │
        │  └────┬───────┘    │  └────┬───────┘    │
        │       │            │       │            │
        │   ┌───┴────┐        │   ┌───┴────┐       │
        │   │ SPLIT  │        │   │ SPLIT  │       │
        │   └───┬────┘        │   └───┬────┘       │
        │       │             │       │             │
        │  ┌────┴──────┐      │      │         ╔═══╝
        │  │ Parallel  │      │   Split    ╚═══╗
        │  │ Attention │      │   Photo    ╔═══╝
        │  │           │      │   Prompt  ╚════╗
        │  │ T→P_text  │      │            
        └──┤ S→P_sketch├──────┴────────────────►Updated Photo Prompt
           │           │          
           │ Concat    │           
           │ & Merge   │           
           └───────────┘           

Key: Photo prompt sees both text AND sketch simultaneously!
```

### 4.2 Deep Layers (4-8): Multi-Source Update Loop

```
Layer 4                   Layer 5  ...              Layer 8
  │                         │                         │
  ├─ Photo Prompt      ├─ Photo Prompt         ├─ Photo Prompt
  │      ↓              │      ↓                │      ↓
  │ ExtractProxy       │ ExtractProxy          │ ExtractProxy
  │      │              │      │                │      │
  │ ┌────┴────┐        │ ┌────┴────┐           │ ┌────┴────┐
  │ │          ↓        │ │          ↓           │ │          ↓
  │ │     ┌────┴─────┐  │ │     ┌────┴─────┐    │ │     ┌────┴─────┐
  ├─┼────→│Photo→Text│  ├─┼────→│Photo→Text│    ├─┼────→│Photo→Text│
  │ │     └────┬─────┘  │ │     └────┬─────┘    │ │     └────┬─────┘
  │ │          ↓        │ │          ↓           │ │          ↓
  │ │     TextPhoto     │ │     TextPhoto        │ │     TextPhoto
  │ │    (EXISTING)     │ │    (EXISTING)        │ │    (EXISTING)
  │ │                   │ │                      │ │
  │ └────→─┐            │ └────→─┐               │ └────→─┐
  │        ↓            │        ↓               │        ↓
  ├─ Sketch Prompt ├─ Sketch Prompt      ├─ Sketch Prompt
  │    (updated   │    (updated from    │    (updated from
  │    from Photo)│     Photo)           │     Photo)
  │        │      │        │              │        │
  │        ├─────→├─ Extract Sketch     │
  │                    │        Proxy    │
  │                    ↓        │        │
  │              ┌──────────┐   │        │
  │         ┌───→│Sketch→   │←──┼┐      │
  ├────────→│    │Text_     │   ││      │
  │         │    │sketch    │   ││      │
  │         │    (NEW)      │   ││      │
  │         │    │ ↓        │   ││      │
  │         │    └──────────┘   ││      │
  │         │       Text_sketch  ││      │
  │         │                     ││      │
  │         └──────────┬──────────┘│      │
  └────────────────────┴───────────┘      │
                                           │
Legend:
  EXISTING: Photo→Text_photo (always there)
  NEW: Photo→Sketch (direct structural guidance)
  NEW: Sketch→Text_sketch (filtered semantic abstraction)

Multi-source effect:
  Text receives from BOTH visual sources
  each providing complementary information!
```

---

## 5. Intuition & Why This Works

### 5.1 Early Layers: Why Split Photo Prompt?

**Problem**: 
- If we feed photo prompts through both text and sketch sequentially, early text updates might overwrite sketch knowledge
- We want BOTH influences equally weighted

**Solution: Split & Parallel**
```
Photo_text_comp   ────→ Attend to Text     ┐
                                            ├─ Merge → Updated Photo
Photo_sketch_comp ────→ Attend to Sketch   ┘

Each component has its own gradient path:
∂L/∂photo_text_comp receives gradient from text attention
∂L/∂photo_sketch_comp receives gradient from sketch attention
→ Explicitly balanced influence!
```

### 5.2 Deep Layers: Why Parallel Photo & Sketch Inputs to Text?

**Intuition: Complementary Information Channels**
```
Photo Branch (Direct): Low-level concrete visual features
  ├─ Texture, color, spatial details
  └─ Updates text_photo: Dense visual semantics

Sketch Branch (Filtered): Mid-level abstract patterns
  ├─ Contours, edges, structural relationships
  └─ Updates text_sketch: Structural grounding

Text Benefits:
  - Gets BOTH dense signal (from photo) AND sparse signal (from sketch)
  - Photo details + Sketch abstractions = Rich representation
  - More robust: complementary info reduces noise
```

vs Chain (Photo → Sketch → Text):
```
Single Linear Path: Photo → Sketch → Text
  ✗ Text only sees sketch's abstraction (loses photo detail)
  ✗ Sequential bottleneck (slower optimization)
  ✗ All photo information compressed through sketch
```

**Why Photo also influences Sketch directly?**
```
Early update (Photo → Sketch):
  - Sketch gets photo's concrete visual structure immediately
  - Doesn't have to wait for text to process
  - Better for fine-grained: early structural guidance
  
Later update (Sketch → Text):
  - Sketch has incorporated photo's details
  - Then refined and sent to text
  - Text sees sketch's distilled + photo-enriched representation
```

### 5.3 When Does This Matter Most?

**Fine-Grained Retrieval (instance matching)**:
- Query: "cat_1.png" (sketch)
- Target: "cat_1_photo.jpg" (specific cat instance)
- Challenge: Same category, but different instance

Traditional (without S↔P):
- Text features might match "cat" category but not instance
- Photo features might match general cat distribution

With Triplet Prompts:
- Photo learns structural quirks of cats from sketch influence
- Sketch learns semantic consistency from photo+text chain
- Better instance discrimination within category

---

## 6. Training Considerations

### 6.1 Gradient Flow & Optimization

```python
# Ensure gradients flow through new components:

New trainable parameters:
1. photo_prompt_text_components (n_ctx/2 × 768 × 4 layers)
2. photo_prompt_sketch_components (n_ctx/2 × 768 × 4 layers)
3. sketch_to_photo_net weights (8 layers)
4. photo_to_sketch_net weights (8 layers)
5. photo_to_sketch_chain weights (5 layers)
6. sketch_to_text_chain weights (5 layers)
7. proxy extraction networks

All these get included in:
- prompt_params group (for lower learning rate)
- Or new group: triplet_params (experimental lr)

Suggested learning rates:
- Base prompts: prompt_lr = 1e-5
- Triplet prompts: triplet_lr = 1e-5 or 1e-6 (slightly slower to stabilize)
- Attention networks: 1e-5
```

### 6.2 Initialization Strategy

```python
# Component initialization importance

For split prompts:
  - Use same initialization as existing prompts (normal_, std=0.02)
  - Or: Initialize from existing visual prompts and perturb slightly
  
For cross-attention networks:
  - Use same CrossPromptAttention as HiCroPL (already stable)
  - But consider:
    * Freeze initial weights for first N epochs?
    * Gradual unfreezing of S↔P networks?
    
For proxy networks:
  - Copy existing AttentionPooling initialization
  - Should be robust
```

### 6.3 Stability Concerns

**Potential Issue 1: Conflicting updates**
- Early layers: Photo gets updated from text AND sketch
- If text/sketch conflict, photo might oscillate

**Solution**:
```python
# Use strict split-query assignment (2+2) to keep sources separated
# q_text comes from first half of photo prompt
# q_sketch comes from second half of photo prompt
# Then concatenate directly without blend coefficient
```

**Potential Issue 2: Chain instability**
- Deep layers: Photo → Sketch → Text
- Updates propagate sequentially, might accumulate errors

**Solution**:
```python
# Keep direct update style (copy_) as existing modal-to-modal update path
# Or: Detach intermediate gradients in early phases
if epoch < warmup_epochs:
    updated_sketch = updated_sketch.detach()  # Don't affect photo
```

### 6.4 Loss Function (Unchanged)

Current loss remains:
```
L_total = L_cross_modal (sketch-photo) + L_ce (text classification)
```

But now features are richer due to:
- Photo learned structural emphasis (from sketch)
- Sketch learned semantic grounding (from text via chain)
→ Same loss, better features → Better convergence

---

## 7. Experimental Validation Plan

### 7.1 Comparison Points

```
Baseline (Current HiCroPL):
  - T ↔ P (text ↔ photo)
  - T ↔ S (text ↔ sketch)
  - No direct S ↔ P

Triplet v1 (Early Split Only):
  - Early layers: Photo split into text/sketch components
  - Deep layers: Original bidirectional (I→T)
  → Test structural influence only

Triplet v2 (Early Split + Deep Chain):
  - Early layers: Photo split (text/sketch)
  - Deep layers: Photo → Sketch → Text chain
  → Full triplet design

Triplet v2.5 (Learnable Alpha):
  - Same as v2, but αᵢ per layer learned
  → Adaptive blending
```

### 7.2 Metrics

**Primary**:
- mAP (category-level retrieval)
- Acc@1, Acc@5, Acc@10 (fine-grained)

**Secondary** (analysis):
- Per-layer feature statistics:
  - Feature norm before/after triplet
  - Gradient magnitude (photo_text vs photo_sketch)
  - Attention weight distributions
  
**Diagnostic**:
- Ablation: Remove S↔P, keep chain → see if chain alone helps
- Ablation: Remove chain, keep split → see if split alone helps
- Component visualization: t-SNE of photo/sketch/text features

### 7.3 Expected Improvements

**Conservative estimate**:
- mAP: +1-2% (fine-grained structure emphasis)
- Acc@1 FG: +2-3% (better instance discrimination)

**Optimistic estimate** (if architecture clicks):
- mAP: +3-5%
- Acc@1 FG: +5-8%

**Risk**: 
- Could hurt if photo tries to be both structural AND semantic
- Might slow training (more components)

---

## 8. Implementation Roadmap

### Phase 1: Minimal Changes
- [ ] Add `photo_prompt_text_components` and `photo_prompt_sketch_components`
- [ ] Add `sketch_to_photo_net` and `photo_to_sketch_net` (use existing CrossPromptAttention)
- [ ] Modify early layer forward pass (layers 0-3) to use split prompts
- [ ] Test: No performance change expected yet (just routing)

### Phase 2: Deep Chain
- [ ] Add `photo_to_sketch_chain` and `sketch_to_text_chain` modules
- [ ] Modify deep layer forward pass (layers 4-8) to chain updates
- [ ] Add proxy extraction for sketch (layers 4-8)
- [ ] Test: Should see stability + slight improvement

### Phase 3: Tuning
- [ ] Hyperparameter sweep (α values, layer split point)
- [ ] Gradient analysis & possible detaching in warmup
- [ ] Comprehensive evaluation

### Phase 4: Variants
- [ ] Learnable alpha per layer
- [ ] Adaptive layer split point
- [ ] Different architectures (e.g., cross-layer instead of split)

---

## 9. Code Integration Points

### Where to modify:

1. **src/hicropl.py**:
   - Extend `CrossModalPromptLearner` → `CrossModalPromptLearner_Triplet`
   - Add new parameters, proxy networks, cross-attention layers
   - Modify `forward()` method

2. **src/model_hicropl.py**:
   - Use `CrossModalPromptLearner_Triplet` instead of `CrossModalPromptLearner` in `CustomCLIP`
   - Rest of pipeline unchanged

3. **experiments/options.py**:
   - New flags:
     - `--triplet_enabled`: bool
     - `--triplet_split_point`: int (default 4)
     - `--triplet_alpha_early`: float (default 0.3)
     - `--triplet_alpha_deep`: float (default 0.3)
     - `--triplet_warmup_epochs`: int (default 5)

4. **experiments/hicropl_prompt.py**:
   - No changes needed (just pass config flags)

### Backward Compatibility:
- Keep `triplet_enabled=False` as default
- Existing code path unchanged if disabled
- Easy A/B testing

---

## 10. Visualization & Analysis

### 10.1 Feature Space Analysis

```python
# After training, compare:
(1) Photo features: In which dimensions does sketch influence show?
    → Compare photo_feat_text_only vs photo_feat_with_sketch
    → Hypothesis: More edge-like activations with sketch

(2) Sketch features: More semantic grounding?
    → Query top-k similar photos (before vs after triplet)
    → Hypothesis: Better semantic consistency

(3) Text features: From chain vs direct?
    → Compare text_from_photo_direct vs text_from_sketch_chain
    → Hypothesis: Chain provides finer gradation
```

### 10.2 Attention Weight Heatmaps

```python
# Visualize learned cross-attentions:
- T→P_text attention: Shape [n_ctx/2, 1] (what text attends to in photo)
- S→P_sketch attention: Shape [n_ctx/2, 1] (what sketch attends to in photo)
- P→S→T chain: Shape [n_ctx, 1] at each step

Hypothesis:
- Early layers: High attention to structural/edge tokens
- Deep layers: High attention to semantic object-part tokens
```

---

## 11. Risk Analysis & Mitigation

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Photo prompt overfitting to split | Medium | Start with α=0.5 (equal blend), decrease gradually |
| Training instability from chain | Medium | Detach intermediate gradients in warmup, use residual |
| Computational overhead | Low | Minimal (reuse existing attention modules) |
| Hyperparameter tuning complexity | Medium | Start with grid search on {α, split_point} |
| Negative transfer from poor sketch knowledge | Medium | Monitor gradients, consider sketch→photo gating |

---

## 12. Future Extensions

Once triplet cross-modal working:

1. **Quaternary prompts** (Add object-part prompts):
   - Text ↔ Object ↔ Photo ↔ Sketch
   
2. **Adaptive routing**:
   - Learn which layer does S↔P matter most
   
3. **Cross-dataset prompts**:
   - Sketch learned from one dataset, photo from another
   
4. **Temporal dynamics**:
   - How does S↔P influence evolve across epochs?

---

## Summary

**Triplet Cross-Modal Design (Parallel Multi-Source)**:
1. ✅ Early Layers: Photo receives BOTH Text AND Sketch influence in parallel
   - Text provides semantic guidance
   - Sketch provides structural guidance
   - Balanced, no sequential overwriting

2. ✅ Deep Layers: Text & Sketch enriched with NEW direct connections
   - Photo → Sketch: Direct visual structure injection (NEW)
   - Photo → Text_photo: Existing visual semantics (unchanged)
   - Sketch → Text_sketch: Filtered structural abstraction (NEW)
   - Result: Text gets dual-channel input for richer representation

3. ✅ Modular & backward-compatible implementation
4. ✅ Potential 2-5% improvement in retrieval metrics

**Key Innovation**: 
- Early: Photo is central hub receiving multiple modality influences
- Deep: Text receives complementary info from photo (direct) and sketch (filtered)
- Photo ↔ Sketch direct cross-talk enables fine-grained structure matching

**Design Philosophy**: Parallel flows over sequential chaining (more robust, richer info)

**Next Step**: Implement Phase 1 minimal changes & validate gradient flow

