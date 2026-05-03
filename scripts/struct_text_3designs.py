"""
Comprehensive analysis of 3 S_text design variants for structural loss.

Design 1: One 2C×2C matrix (sketch-sketch, photo-photo, cross-modal blocks)
Design 2: Two separate C×C matrices (S_text_sketch and S_text_photo)
Design 3: One shared C×C matrix (single template for both modalities)

For each design, we:
1. Compute text geometry matrix(ces)
2. Compute visual geometry matrix(ces) from samples
3. Measure alignment (Pearson, Spearman correlation)
4. Analyze block structure (Design 1 only)
5. Measure cross-modal consistency
6. Produce comparison report and visualizations

Usage:
python scripts/struct_text_3designs.py \
    --image-root /path/to/Sketchy \
    --out-dir results/3designs_analysis \
    --max-classes 50 --samples-per-class 5 \
    --device cuda
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

from src.clip import clip as _clip

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except:
    KMeans = None

try:
    from scipy.stats import pearsonr, spearmanr
except:
    pearsonr = None
    spearmanr = None

try:
    import seaborn as sns
except:
    sns = None


def clean_name(name):
    """Convert class name: airplane -> Airplane"""
    return name.replace('_', ' ').title()


def load_classnames(path):
    """Load from file or auto-detect from folder."""
    if os.path.isdir(path):
        p = Path(path)
        photo_dir = p / 'photo'
        sketch_dir = p / 'sketch'
        
        if photo_dir.exists() and sketch_dir.exists():
            candidates = sorted([c.name for c in photo_dir.iterdir() 
                               if c.is_dir() and not c.name.startswith('.')])
            if candidates:
                print(f'[INFO] Auto-detected {len(candidates)} classes from photo/ subfolder')
                return candidates
        
        candidates = sorted([item.name for item in p.iterdir() 
                            if item.is_dir() and not item.name.startswith('.')])
        candidates = [c for c in candidates if c not in ['photo', 'sketch']]
        
        if candidates:
            print(f'[INFO] Auto-detected {len(candidates)} classes (direct subfolders)')
            return sorted(candidates)
        raise RuntimeError(f'No classes found in {path}')
    else:
        with open(path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines


def compute_text_embeddings_per_template(model, device, classnames, templates):
    """
    Compute text embeddings for each template separately.
    Returns dict: {template_str: tensor of shape [num_classes, embedding_dim]}
    """
    model.eval()
    result = {}
    
    with torch.no_grad():
        for tmpl in templates:
            texts = [tmpl.format(clean_name(c)) for c in classnames]
            tokenized = _clip.tokenize(texts).to(device)
            emb = model.encode_text(tokenized)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            result[tmpl] = emb.cpu()
    
    return result


def compute_ensemble_embeddings(per_template_embeds):
    """Average embeddings across templates."""
    embeds_list = list(per_template_embeds.values())
    stacked = torch.stack(embeds_list, dim=0)  # [num_templates, num_classes, dim]
    ensemble = stacked.mean(dim=0)  # [num_classes, dim]
    ensemble = ensemble / ensemble.norm(dim=-1, keepdim=True)
    return ensemble


def sample_images_per_class(image_root, classnames, max_classes, samples_per_class, modality):
    """Sample images for a specific modality."""
    image_root = Path(image_root)
    search_root = image_root / modality
    
    if not search_root.exists():
        return []
    
    available = []
    for cname in classnames[:max_classes]:
        folder = search_root / cname
        if folder.exists() and folder.is_dir():
            files = list(folder.glob('*'))
            files = [p for p in files if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            if files:
                available.append((cname, files))
    
    if not available:
        return []
    
    sampled = []
    for cname, files in available[:max_classes]:
        if len(files) <= samples_per_class:
            picks = files
        else:
            picks = list(np.random.choice(files, samples_per_class, replace=False))
        sampled.append((cname, picks))
    
    return sampled


def compute_visual_class_centroids(model, preprocess, device, sampled, batch_size=32):
    """Compute visual feature centroid for each class."""
    model.eval()
    centroids = {}
    from PIL import Image
    
    with torch.no_grad():
        for cname, files in sampled:
            imgs = []
            for p in files:
                try:
                    im = Image.open(p).convert('RGB')
                    imgs.append(preprocess(im))
                except:
                    continue
            
            if not imgs:
                continue
            
            img_batch = torch.stack(imgs).to(device)
            feats = model.encode_image(img_batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            centroid = feats.mean(dim=0)
            centroid = centroid / centroid.norm()
            centroids[cname] = centroid.cpu().numpy()
    
    return centroids


def compute_similarity_matrix(embeds):
    """Compute similarity matrix from embeddings."""
    if isinstance(embeds, torch.Tensor):
        embeds = embeds.numpy()
    S = np.matmul(embeds, embeds.T)
    return S


def flatten_upper_tri(mat):
    """Flatten upper triangle (excluding diagonal)."""
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def compute_block_stats(S_2C, num_classes):
    """
    For Design 1: Analyze 4 blocks of 2C×2C matrix.
    Returns dict with stats for each block.
    """
    C = num_classes
    
    # Extract blocks
    S_sketch_sketch = S_2C[:C, :C]  # top-left
    S_photo_photo = S_2C[C:, C:]    # bottom-right
    S_sketch_photo = S_2C[:C, C:]   # top-right (cross-modal)
    S_photo_sketch = S_2C[C:, :C]   # bottom-left (cross-modal)
    
    def block_stats(block, name):
        diag_mask = np.eye(block.shape[0], dtype=bool)
        offdiag = block[np.triu_indices(block.shape[0], k=1)]
        return {
            f'{name}_mean': float(offdiag.mean()),
            f'{name}_std': float(offdiag.std()),
            f'{name}_min': float(offdiag.min()),
            f'{name}_max': float(offdiag.max()),
        }
    
    stats = {}
    stats.update(block_stats(S_sketch_sketch, 'sketch_sketch'))
    stats.update(block_stats(S_photo_photo, 'photo_photo'))
    
    # For cross-modal blocks, take all entries (not just upper triangle)
    S_sketch_photo_flat = S_sketch_photo.flatten()
    S_photo_sketch_flat = S_photo_sketch.flatten()
    
    stats.update({
        'sketch_photo_mean': float(S_sketch_photo_flat.mean()),
        'sketch_photo_std': float(S_sketch_photo_flat.std()),
        'photo_sketch_mean': float(S_photo_sketch_flat.mean()),
        'photo_sketch_std': float(S_photo_sketch_flat.std()),
    })
    
    return stats


def compute_correlations(S_text, S_visual_sketch, S_visual_photo, classnames_subset, classnames_full):
    """
    Compute Pearson/Spearman correlation between text and visual matrices.
    Handles different sizes by subsetting.
    """
    # Map classnames_subset to indices in classnames_full
    idx_map = {name: i for i, name in enumerate(classnames_full)}
    indices = [idx_map[n] for n in classnames_subset]
    
    # Subset S_text
    S_text_sub = S_text[np.ix_(indices, indices)]
    
    # Flatten upper triangle
    v_text = flatten_upper_tri(S_text_sub)
    v_visual_sketch = flatten_upper_tri(S_visual_sketch)
    v_visual_photo = flatten_upper_tri(S_visual_photo)
    
    results = {}
    
    if pearsonr and spearmanr:
        r_sketch, p_sketch = pearsonr(v_text, v_visual_sketch)
        r_photo, p_photo = pearsonr(v_text, v_visual_photo)
        rho_sketch, _ = spearmanr(v_text, v_visual_sketch)
        rho_photo, _ = spearmanr(v_text, v_visual_photo)
    else:
        r_sketch = np.corrcoef(v_text, v_visual_sketch)[0, 1]
        r_photo = np.corrcoef(v_text, v_visual_photo)[0, 1]
        rho_sketch = rho_photo = 0.0
    
    results = {
        'pearson_sketch': float(r_sketch),
        'pearson_photo': float(r_photo),
        'spearman_sketch': float(rho_sketch),
        'spearman_photo': float(rho_photo),
    }
    
    return results, (v_text, v_visual_sketch, v_visual_photo)


def compute_cross_modal_consistency(centroids_sketch, centroids_photo, classnames_subset):
    """
    For Design 3: Measure average cosine similarity between sketch and photo 
    centroids of same class.
    """
    scores = []
    for cname in classnames_subset:
        if cname in centroids_sketch and cname in centroids_photo:
            s = centroids_sketch[cname]
            p = centroids_photo[cname]
            sim = np.dot(s, p)  # already normalized
            scores.append(sim)
    
    if scores:
        return float(np.mean(scores))
    return 0.0


def visualize_designs(design_results, out_dir):
    """Create comprehensive visualizations comparing 3 designs."""
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Block structure for Design 1
    if 'design1_blocks' in design_results:
        blocks = design_results['design1_blocks']
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        block_names = ['sketch_sketch', 'photo_photo', 'sketch_photo', 'photo_sketch']
        for idx, (ax, name) in enumerate(zip(axes.flat, block_names)):
            mean_val = blocks.get(f'{name}_mean', 0)
            std_val = blocks.get(f'{name}_std', 0)
            ax.bar(['mean', 'std'], [mean_val, std_val], color=['blue', 'red'])
            ax.set_title(f'Design 1: {name.replace("_", "-")}')
            ax.set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'design1_block_structure.png'), dpi=100)
        plt.close()
    
    # 2. Correlation comparison bar chart
    if 'correlations' in design_results:
        corrs = design_results['correlations']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        designs = list(corrs.keys())
        metrics = ['pearson_sketch', 'pearson_photo']
        
        x = np.arange(len(designs))
        width = 0.35
        
        sketch_vals = [corrs[d].get('pearson_sketch', 0) for d in designs]
        photo_vals = [corrs[d].get('pearson_photo', 0) for d in designs]
        
        ax.bar(x - width/2, sketch_vals, width, label='vs S_visual_sketch')
        ax.bar(x + width/2, photo_vals, width, label='vs S_visual_photo')
        
        ax.set_ylabel('Pearson r')
        ax.set_title('Correlation: S_text vs Visual Geometry (3 Designs)')
        ax.set_xticks(x)
        ax.set_xticklabels(designs)
        ax.legend()
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'correlation_comparison.png'), dpi=100)
        plt.close()
    
    # 3. Heatmaps for Design 1, 2, 3
    for design_name in ['design1', 'design2_sketch', 'design2_photo', 'design3']:
        if f'{design_name}_matrix' in design_results:
            mat = design_results[f'{design_name}_matrix']
            if mat.size > 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(mat, cmap='viridis', vmin=-1, vmax=1)
                ax.set_title(f'S_text Heatmap: {design_name.replace("_", " ").title()}')
                plt.colorbar(im, ax=ax)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'{design_name}_heatmap.png'), dpi=100)
                plt.close()
    
    # 4. Scatter plots of correlations
    if 'scatter_data' in design_results:
        scatter = design_results['scatter_data']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, design_name in enumerate(['design1', 'design2', 'design3']):
            if design_name in scatter:
                v_text, v_visual_sketch, v_visual_photo = scatter[design_name]
                
                ax = axes[idx]
                ax.scatter(v_text, v_visual_sketch, s=4, alpha=0.5, label='sketch', color='blue')
                ax.scatter(v_text, v_visual_photo, s=4, alpha=0.5, label='photo', color='orange')
                ax.set_xlabel('S_text (upper tri)')
                ax.set_ylabel('S_visual (upper tri)')
                ax.set_title(f'{design_name.replace("_", " ").title()}')
                ax.legend()
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'correlation_scatter_3designs.png'), dpi=100)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare 3 S_text design variants')
    parser.add_argument('--classnames-file', default=None)
    parser.add_argument('--image-root', default=None)
    parser.add_argument('--out-dir', default='results/3designs_analysis')
    parser.add_argument('--model', default='ViT-B/32')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-classes', type=int, default=50)
    parser.add_argument('--samples-per-class', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load classnames
    if args.classnames_file:
        classnames = load_classnames(args.classnames_file)
    elif args.image_root:
        classnames = load_classnames(args.image_root)
    else:
        raise RuntimeError('Must provide --classnames-file or --image-root')
    
    K = min(len(classnames), args.max_classes)
    classnames = classnames[:K]
    print(f'[INFO] Using {K} classes')
    
    device = torch.device(args.device)
    print(f'[INFO] Loading CLIP model {args.model}')
    clip_model, preprocess = _clip.load(args.model, device=device, jit=False)
    
    # ===== DESIGN SETUP =====
    
    # Templates for Design 1, 2
    template_photo = "a photo of a {}"
    template_sketch = "a sketch of a {}"
    template_shared = "a photo or a sketch of a {}"
    
    print(f'\n[INFO] Computing text embeddings per template...')
    per_template = compute_text_embeddings_per_template(
        clip_model, device, classnames, 
        [template_photo, template_sketch, template_shared]
    )
    
    # ===== DESIGN 1: 2C×2C Matrix =====
    print(f'\n[DESIGN 1] Building 2C×2C matrix (sketch + photo stacked)...')
    emb_sketch = per_template[template_sketch]
    emb_photo = per_template[template_photo]
    emb_2C = torch.cat([emb_sketch, emb_photo], dim=0)  # [2C, D]
    S_text_2C = compute_similarity_matrix(emb_2C)
    
    design1_blocks = compute_block_stats(S_text_2C, K)
    print(f'[DESIGN 1] Block stats: {design1_blocks}')
    
    # ===== DESIGN 2: Two separate C×C matrices =====
    print(f'\n[DESIGN 2] Building two C×C matrices separately...')
    S_text_sketch = compute_similarity_matrix(emb_sketch)
    S_text_photo = compute_similarity_matrix(emb_photo)
    
    if pearsonr:
        r_sketch_sketch, _ = pearsonr(flatten_upper_tri(S_text_sketch), 
                                       flatten_upper_tri(S_text_sketch))
        r_photo_photo, _ = pearsonr(flatten_upper_tri(S_text_photo), 
                                     flatten_upper_tri(S_text_photo))
        r_cross, _ = pearsonr(flatten_upper_tri(S_text_sketch), 
                              flatten_upper_tri(S_text_photo))
        print(f'[DESIGN 2] Correlation sketch-sketch vs photo-photo: {r_cross:.4f}')
    
    # ===== DESIGN 3: One shared C×C matrix =====
    print(f'\n[DESIGN 3] Building one C×C shared matrix...')
    emb_shared = per_template[template_shared]
    S_text_shared = compute_similarity_matrix(emb_shared)
    
    # ===== VISUAL FEATURES (if image_root provided) =====
    design_results = {
        'design1_blocks': design1_blocks,
        'design1_matrix': S_text_2C,
        'design2_sketch_matrix': S_text_sketch,
        'design2_photo_matrix': S_text_photo,
        'design3_matrix': S_text_shared,
        'correlations': {},
        'scatter_data': {},
    }
    
    if args.image_root:
        print(f'\n[INFO] Computing visual centroids for sketch and photo...')
        
        sampled_sketch = sample_images_per_class(args.image_root, classnames, K, 
                                                 args.samples_per_class, 'sketch')
        sampled_photo = sample_images_per_class(args.image_root, classnames, K, 
                                                args.samples_per_class, 'photo')
        
        if sampled_sketch:
            print(f'[INFO] Sampled {len(sampled_sketch)} sketch classes')
            centroids_sketch = compute_visual_class_centroids(clip_model, preprocess, device, sampled_sketch)
            names_sketch = [c for c, _ in sampled_sketch if c in centroids_sketch]
            C_sketch = np.stack([centroids_sketch[n] for n in names_sketch], axis=0)
            S_visual_sketch = compute_similarity_matrix(C_sketch)
        else:
            S_visual_sketch = None
            names_sketch = []
            centroids_sketch = {}
        
        if sampled_photo:
            print(f'[INFO] Sampled {len(sampled_photo)} photo classes')
            centroids_photo = compute_visual_class_centroids(clip_model, preprocess, device, sampled_photo)
            names_photo = [c for c, _ in sampled_photo if c in centroids_photo]
            C_photo = np.stack([centroids_photo[n] for n in names_photo], axis=0)
            S_visual_photo = compute_similarity_matrix(C_photo)
        else:
            S_visual_photo = None
            names_photo = []
            centroids_photo = {}
        
        # Compute correlations for all 3 designs
        common_names = sorted(set(names_sketch) & set(names_photo))
        print(f'[INFO] Common classes for correlation: {len(common_names)}')
        
        if S_visual_sketch is not None and S_visual_photo is not None and common_names:
            
            # Design 1 correlation
            print(f'\n[DESIGN 1] Computing correlations...')
            # For 2C matrix, we need to subset it differently
            idx_sketch = [i for i, n in enumerate(names_sketch) if n in common_names]
            idx_photo = [i for i, n in enumerate(names_photo) if n in common_names]
            S_visual_sketch_sub = S_visual_sketch[np.ix_(idx_sketch, idx_sketch)]
            S_visual_photo_sub = S_visual_photo[np.ix_(idx_photo, idx_photo)]
            
            idx_all = []
            for i, c in enumerate(classnames):
                if c in common_names:
                    idx_all.append(i)
            
            # Extract corners from 2C matrix
            C = len(classnames)
            S_2C_sketch_corner = S_text_2C[:C, :C]
            S_2C_photo_corner = S_text_2C[C:, C:]
            S_2C_sketch_corner_sub = S_2C_sketch_corner[np.ix_(idx_all, idx_all)]
            S_2C_photo_corner_sub = S_2C_photo_corner[np.ix_(idx_all, idx_all)]
            
            if pearsonr and spearmanr:
                r1_s, _ = pearsonr(flatten_upper_tri(S_2C_sketch_corner_sub), 
                                   flatten_upper_tri(S_visual_sketch_sub))
                r1_p, _ = pearsonr(flatten_upper_tri(S_2C_photo_corner_sub), 
                                   flatten_upper_tri(S_visual_photo_sub))
            else:
                r1_s = np.corrcoef(flatten_upper_tri(S_2C_sketch_corner_sub), 
                                   flatten_upper_tri(S_visual_sketch_sub))[0, 1]
                r1_p = np.corrcoef(flatten_upper_tri(S_2C_photo_corner_sub), 
                                   flatten_upper_tri(S_visual_photo_sub))[0, 1]
            
            design_results['correlations']['design1'] = {
                'pearson_sketch': float(r1_s),
                'pearson_photo': float(r1_p),
            }
            print(f'[DESIGN 1] Correlation: sketch={r1_s:.4f}, photo={r1_p:.4f}')
            
            # Design 2 correlation
            print(f'\n[DESIGN 2] Computing correlations...')
            if pearsonr and spearmanr:
                r2_s, _ = pearsonr(flatten_upper_tri(S_text_sketch[np.ix_(idx_all, idx_all)]), 
                                   flatten_upper_tri(S_visual_sketch_sub))
                r2_p, _ = pearsonr(flatten_upper_tri(S_text_photo[np.ix_(idx_all, idx_all)]), 
                                   flatten_upper_tri(S_visual_photo_sub))
            else:
                r2_s = np.corrcoef(flatten_upper_tri(S_text_sketch[np.ix_(idx_all, idx_all)]), 
                                   flatten_upper_tri(S_visual_sketch_sub))[0, 1]
                r2_p = np.corrcoef(flatten_upper_tri(S_text_photo[np.ix_(idx_all, idx_all)]), 
                                   flatten_upper_tri(S_visual_photo_sub))[0, 1]
            
            design_results['correlations']['design2'] = {
                'pearson_sketch': float(r2_s),
                'pearson_photo': float(r2_p),
            }
            print(f'[DESIGN 2] Correlation: sketch={r2_s:.4f}, photo={r2_p:.4f}')
            
            # Design 3 correlation
            print(f'\n[DESIGN 3] Computing correlations...')
            S_text_shared_sub = S_text_shared[np.ix_(idx_all, idx_all)]
            if pearsonr and spearmanr:
                r3_s, _ = pearsonr(flatten_upper_tri(S_text_shared_sub), 
                                   flatten_upper_tri(S_visual_sketch_sub))
                r3_p, _ = pearsonr(flatten_upper_tri(S_text_shared_sub), 
                                   flatten_upper_tri(S_visual_photo_sub))
            else:
                r3_s = np.corrcoef(flatten_upper_tri(S_text_shared_sub), 
                                   flatten_upper_tri(S_visual_sketch_sub))[0, 1]
                r3_p = np.corrcoef(flatten_upper_tri(S_text_shared_sub), 
                                   flatten_upper_tri(S_visual_photo_sub))[0, 1]
            
            design_results['correlations']['design3'] = {
                'pearson_sketch': float(r3_s),
                'pearson_photo': float(r3_p),
            }
            print(f'[DESIGN 3] Correlation: sketch={r3_s:.4f}, photo={r3_p:.4f}')
            
            # Cross-modal consistency for Design 3
            consistency_score = compute_cross_modal_consistency(centroids_sketch, centroids_photo, common_names)
            design_results['design3_consistency'] = consistency_score
            print(f'[DESIGN 3] Cross-modal consistency (avg cosine sim): {consistency_score:.4f}')
            
            # Store scatter data for visualization
            design_results['scatter_data']['design1'] = (
                flatten_upper_tri(S_2C_sketch_corner_sub),
                flatten_upper_tri(S_visual_sketch_sub),
                flatten_upper_tri(S_visual_photo_sub),
            )
            design_results['scatter_data']['design2'] = (
                flatten_upper_tri(S_text_sketch[np.ix_(idx_all, idx_all)]),
                flatten_upper_tri(S_visual_sketch_sub),
                flatten_upper_tri(S_visual_photo_sub),
            )
            design_results['scatter_data']['design3'] = (
                flatten_upper_tri(S_text_shared_sub),
                flatten_upper_tri(S_visual_sketch_sub),
                flatten_upper_tri(S_visual_photo_sub),
            )
    
    # Save results
    print(f'\n[INFO] Saving results...')
    
    # Save matrices
    np.save(os.path.join(args.out_dir, 'S_text_design1_2C.npy'), S_text_2C)
    np.save(os.path.join(args.out_dir, 'S_text_design2_sketch.npy'), S_text_sketch)
    np.save(os.path.join(args.out_dir, 'S_text_design2_photo.npy'), S_text_photo)
    np.save(os.path.join(args.out_dir, 'S_text_design3_shared.npy'), S_text_shared)
    
    # Save metrics as JSON
    metrics_json = {
        'design1_blocks': design_results['design1_blocks'],
        'correlations': design_results['correlations'],
    }
    if 'design3_consistency' in design_results:
        metrics_json['design3_cross_modal_consistency'] = design_results['design3_consistency']
    
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f'[INFO] Metrics saved to {os.path.join(args.out_dir, "metrics.json")}')
    
    # Visualizations
    print(f'[INFO] Creating visualizations...')
    visualize_designs(design_results, args.out_dir)
    
    # Print summary
    print(f'\n{"="*60}')
    print(f'SUMMARY: 3 Design Comparison')
    print(f'{"="*60}')
    print(f'Design 1 (2C×2C matrix):')
    print(f'  - Block stats: {design_results["design1_blocks"]}')
    if 'design1' in design_results['correlations']:
        print(f'  - Corr sketch: {design_results["correlations"]["design1"]["pearson_sketch"]:.4f}')
        print(f'  - Corr photo: {design_results["correlations"]["design1"]["pearson_photo"]:.4f}')
    
    print(f'\nDesign 2 (Two C×C matrices):')
    if 'design2' in design_results['correlations']:
        print(f'  - Corr sketch: {design_results["correlations"]["design2"]["pearson_sketch"]:.4f}')
        print(f'  - Corr photo: {design_results["correlations"]["design2"]["pearson_photo"]:.4f}')
    
    print(f'\nDesign 3 (One shared C×C matrix):')
    if 'design3' in design_results['correlations']:
        print(f'  - Corr sketch: {design_results["correlations"]["design3"]["pearson_sketch"]:.4f}')
        print(f'  - Corr photo: {design_results["correlations"]["design3"]["pearson_photo"]:.4f}')
    if 'design3_consistency' in design_results:
        print(f'  - Cross-modal consistency: {design_results["design3_consistency"]:.4f}')
    
    print(f'\n[INFO] Results saved to {args.out_dir}')


if __name__ == '__main__':
    main()
