# 3 S_text Design Variant Evaluation

## Overview

Comprehensive analysis of 3 alternative designs for structural loss matrix (S_text) in HiCroPL SBIR:

1. **Design 1**: One 2C×2C matrix combining sketch and photo embeddings
2. **Design 2**: Two separate C×C matrices (modality-specific)  
3. **Design 3**: One shared C×C matrix (balanced template)

## Workflow

### Step 1: Run Analysis Script

```bash
cd /Users/tranduc/nckh/ZS-SBIR/Sketch_VLM

python scripts/struct_text_3designs.py \
  --image-root /path/to/Sketchy \
  --out-dir results/3designs_analysis \
  --max-classes 50 \
  --samples-per-class 5 \
  --device cuda
```

**Inputs:**
- `--image-root`: Path to Sketchy dataset (expects `photo/` and `sketch/` subdirs)
- `--max-classes`: Number of classes to sample (default 50)
- `--samples-per-class`: Images per class for visual centroids (default 5, use 3-5 for speed)
- `--device`: cuda or cpu

**Outputs in `results/3designs_analysis/`:**
- `S_text_design1_2C.npy`: 2C×2C matrix
- `S_text_design2_sketch.npy`: C×C sketch-specific
- `S_text_design2_photo.npy`: C×C photo-specific  
- `S_text_design3_shared.npy`: C×C balanced
- `metrics.json`: Detailed metrics
- `design1_block_structure.png`: Block analysis heatmap
- `correlation_comparison.png`: Bar chart comparing correlations
- `correlation_scatter_3designs.png`: Scatter plots of alignment
- `design*_heatmap.png`: Heatmaps for each design

### Step 2: Run Evaluation Script

```bash
python scripts/evaluate_3designs.py \
  --metrics-file results/3designs_analysis/metrics.json \
  --out-dir results/3designs_analysis
```

**Output:**
- Console report with findings, concerns, recommendations
- `evaluation_report.json`: Full structured evaluation

## What Each Design Measures

### Design 1: 2C×2C Matrix (Explicit Cross-Modal)

**Block Structure (4 blocks):**
```
Sketch-Sketch (C×C)  │  Sketch-Photo (C×C)
────────────────────┼────────────────────
Photo-Sketch (C×C)  │  Photo-Photo (C×C)
```

**Key Metrics:**
- `sketch_sketch_mean`: Within-sketch similarity
- `photo_photo_mean`: Within-photo similarity
- `sketch_photo_mean`: Cross-modal similarity (sketch→photo)
- Block stds: Discriminativeness (low std = flat/bad)

**When to use:**
- Need fine-grained control of sketch-photo interaction
- Want to ablate cross-modal blocks separately
- Willing to monitor 4 loss terms during training

---

### Design 2: Two Separate C×C Matrices (Modality-Specific)

**Matrices:**
- `S_text_sketch = encode("a sketch of a {}")` similarity
- `S_text_photo = encode("a photo of a {}")` similarity

**Key Metrics:**
- `pearson_sketch`: Alignment of sketch-specific matrix with sketch visual features
- `pearson_photo`: Alignment of photo-specific matrix with photo visual features
- Difference between the two: How different are modalities?

**When to use:**
- Sketch and photo have different visual characteristics
- Want independent loss tracking per modality
- Can selectively disable loss for problematic modality

---

### Design 3: Shared C×C Matrix (Simple)

**Matrix:**
- `S_text_shared = encode("a photo or a sketch of a {}")` similarity

**Key Metrics:**
- `pearson_sketch`: How well does shared matrix align with sketch visual space?
- `pearson_photo`: How well does shared matrix align with photo visual space?
- `cross_modal_consistency`: Average cosine(sketch_centroid, photo_centroid) for same class
  - High (>0.5): sketch and photo embeddings naturally align ✓
  - Low (<0.3): they diverge, shared matrix may not hold ✗

**When to use:**
- Want simplest, most interpretable design
- Sketch and photo have similar visual geometry
- Cross-modal consistency is already good

---

## Interpreting Results

### Example Scenario 1: Design 2 Wins

```
Correlation Comparison:
  Design 1: avg=0.12
  Design 2: avg=0.45 ← BEST
  Design 3: avg=0.18

Modality Analysis (Design 2):
  - Pearson vs sketch visual: 0.50
  - Pearson vs photo visual: 0.40  
  → Modalities differ (diff=0.10), separate matrices justified
```

**Recommendation:** Use Design 2
- Log L_struct_sketch and L_struct_photo separately
- Can ablate: train with photo only, then only sketch
- Photo template more aligned; consider it as primary

---

### Example Scenario 2: Design 3 Wins

```
Correlation Comparison:
  Design 1: avg=0.22
  Design 2: avg=0.25
  Design 3: avg=0.48 ← BEST

Cross-Modal Consistency (Design 3):
  sketch-photo similarity: 0.68  → HIGH
  
Block Similarity (Design 1):
  sketch_sketch: 0.40
  photo_photo: 0.38
  cross_modal: 0.35  → All similar
```

**Recommendation:** Use Design 3
- Shared matrix works because sketch and photo naturally align in text space
- Simplest, fewest assumptions
- But monitor zero-shot: is sketch→photo as good as photo→sketch?

---

### Example Scenario 3: All Low (Design 1,2,3 all <0.2)

```
Correlation Comparison:
  Design 1: avg=0.08
  Design 2: avg=0.10
  Design 3: avg=0.09
  
All very low!
```

**Recommendation:** **STOP — rethink the approach**
- Text-visual gap is fundamental for this dataset
- No template or design variant solves it
- Consider:
  - Use text geometry for initialization only, then fine-tune with visual loss
  - Disable structural loss (λ_struct=0)
  - Use visual centroids (from previous batches) instead of text as anchor
  - Question whether "geometry anchor" is the right lever for ZS-SBIR

---

## Key Metrics to Compare

| Metric | Design 1 | Design 2 | Design 3 |
|--------|----------|----------|----------|
| **Pearson (sketch)** | corr_s1 | corr_s2 | corr_s3 |
| **Pearson (photo)** | corr_p1 | corr_p2 | corr_p3 |
| **Complexity** | High | Medium | Low |
| **Interpretability** | Lower (4 blocks) | Good (2 matrices) | Best (1 matrix) |
| **Ablation** | Hard (coupled blocks) | Easy (per modality) | Limited |

---

## Next Steps After Evaluation

### If Best Design is Chosen:

1. **Implement in src/model_hicropl.py**
   - Modify `_initialize_struct_text_geometry()`
   - Update `_compute_structural_loss()` to use chosen design
   - Add per-block or per-modality loss logging

2. **Training Ablation**
   - Train with chosen design + λ_struct=0.01
   - Compare with baseline (λ_struct=0)
   - Check mAP improvement, convergence speed, stability

3. **Monitor During Training**
   - Log alignment quality (Pearson r) at validation time
   - If correlation decreases mid-training → structural loss is harmful
   - Track block losses separately (Design 1) or modality losses (Design 2)

4. **Zero-Shot Evaluation**
   - Measure sketch→photo and photo→sketch mAP separately
   - Check if one direction regresses (may indicate asymmetry)
   - Use Design 2 if significant imbalance observed

---

## Files Generated

```
results/3designs_analysis/
├── S_text_design1_2C.npy              # 2C×2C matrix
├── S_text_design2_sketch.npy          # C×C for sketch
├── S_text_design2_photo.npy           # C×C for photo
├── S_text_design3_shared.npy          # C×C shared
├── metrics.json                       # Raw metrics
├── evaluation_report.json             # Full evaluation
├── design1_block_structure.png        # Block analysis
├── correlation_comparison.png         # Bar chart
├── correlation_scatter_3designs.png   # Scatter plots
├── design1_heatmap.png                # Heatmap 1
├── design2_sketch_heatmap.png         # Heatmap 2
├── design2_photo_heatmap.png          # Heatmap 2
└── design3_heatmap.png                # Heatmap 3
```

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'src.clip'"**
- Run from repo root: `cd /Users/tranduc/nckh/ZS-SBIR/Sketch_VLM`

**"No classes found in /path/to/Sketchy"**
- Verify path has `photo/` and `sketch/` subdirs
- Check: `ls -la /path/to/Sketchy/photo/ | head -5`

**"No photo/sketch images found"**
- Script will gracefully skip that modality
- May return partial results (only one modality analyzed)
- Check image file extensions: .jpg, .png, .jpeg, .bmp are supported

**Low correlations for all designs**
- This is expected for some datasets/templates
- Indicates text-visual space mismatch
- Structural loss may not be suitable; consider disabling
