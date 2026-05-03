import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np
from scipy.stats import describe
from sklearn.metrics.pairwise import cosine_similarity

from src.clip import clip as _clip

try:
    from scipy.stats import pearsonr
except:
    pearsonr = None

def clean_name(name):
    return name.replace('_', ' ').title()

def load_classnames(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

def compute_text_embeddings_per_template(model, device, classnames, templates):
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

def compute_similarity_matrix(embeds):
    if isinstance(embeds, torch.Tensor):
        embeds = embeds.numpy()
    S = np.matmul(embeds, embeds.T)
    return S

def flatten_upper_tri(mat):
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]

def compute_block_stats(S_2C, num_classes):
    C = num_classes
    S_sketch_sketch = S_2C[:C, :C]
    S_photo_photo = S_2C[C:, C:]
    S_sketch_photo = S_2C[:C, C:]
    
    stats = {
        'sketch_sketch_mean': float(flatten_upper_tri(S_sketch_sketch).mean()),
        'sketch_sketch_std': float(flatten_upper_tri(S_sketch_sketch).std()),
        'photo_photo_mean': float(flatten_upper_tri(S_photo_photo).mean()),
        'photo_photo_std': float(flatten_upper_tri(S_photo_photo).std()),
        'cross_modal_mean': float(S_sketch_photo.flatten().mean()),
        'cross_modal_std': float(S_sketch_photo.flatten().std()),
    }
    return stats

def visualize_initial_S_text(matrices, block_stats, design2_corr, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Visualization 1: Heatmaps ---
    # So sánh trực quan cấu trúc tổng thể của các ma trận
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    heatmap_data = {
        'Design 1 (2C)': matrices['design1'],
        'Design 2 (Sketch)': matrices['design2_sketch'],
        'Design 2 (Photo)': matrices['design2_photo'],
        'Design 3 (Shared)': matrices['design3'],
    }
    for ax, (title, mat) in zip(axes, heatmap_data.items()):
        im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle('Visualization 1: S_text Heatmaps Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, '1_heatmaps_comparison.png'), dpi=120)
    plt.close()

    # --- Visualization 2: Histograms of Off-Diagonal Values ---
    # Cho thấy sự phân bố giá trị similarity, ma trận "phẳng" hay "đa dạng"
    fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
    hist_data = {
        'D1 Sketch-Sketch': flatten_upper_tri(matrices['design1'][:matrices['design2_sketch'].shape[0], :matrices['design2_sketch'].shape[0]]),
        'D2 Sketch': flatten_upper_tri(matrices['design2_sketch']),
        'D2 Photo': flatten_upper_tri(matrices['design2_photo']),
        'D3 Shared': flatten_upper_tri(matrices['design3']),
    }
    for ax, (title, data) in zip(axes, hist_data.items()):
        ax.hist(data, bins=50, range=(0, 1))
        ax.set_title(title)
        ax.set_xlabel('Similarity')
        ax.grid(axis='y', alpha=0.5)
    axes[0].set_ylabel('Frequency')
    plt.suptitle('Visualization 2: Distribution of Off-Diagonal Similarities', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, '2_histograms_distribution.png'), dpi=120)
    plt.close()

    # --- Visualization 3: Key Metrics Bar Chart ---
    # Tóm tắt các chỉ số quan trọng nhất để so sánh nhanh
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data for plotting
    labels = ['D1 Sketch Mean', 'D1 Photo Mean', 'D1 Cross Mean', 'D2 Sketch-Photo Corr']
    values = [
        block_stats.get('sketch_sketch_mean', 0),
        block_stats.get('photo_photo_mean', 0),
        block_stats.get('cross_modal_mean', 0),
        design2_corr
    ]
    colors = ['blue', 'blue', 'green', 'red']
    
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Value')
    ax.set_title('Visualization 3: Key Structural Metrics')
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom') # va: vertical alignment

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '3_key_metrics_summary.png'), dpi=120)
    plt.close()

def convert_to_native_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description='Analyze initial S_text matrix designs')
    parser.add_argument('--classnames-file', required=True, help='Path to file with class names, one per line.')
    parser.add_argument('--out-dir', default='results/S_text_initial_analysis')
    parser.add_argument('--model', default='ViT-B/32')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-classes', type=int, default=125)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    classnames = load_classnames(args.classnames_file)[:args.max_classes]
    K = len(classnames)
    print(f'[INFO] Analyzing {K} classes from {args.classnames_file}')
    
    device = torch.device(args.device)
    clip_model, _ = _clip.load(args.model, device=device, jit=False)
    
    template_photo = "a photo of a {}"
    template_sketch = "a sketch of a {}"
    template_shared = "a photo or a sketch of a {}"
    
    print('[INFO] Computing text embeddings...')
    per_template = compute_text_embeddings_per_template(
        clip_model, device, classnames, 
        [template_photo, template_sketch, template_shared]
    )
    
    # --- Design 1: 2C×2C Matrix ---
    print('[INFO] Building Design 1: 2C×2C Matrix...')
    emb_sketch = per_template[template_sketch]
    emb_photo = per_template[template_photo]
    emb_2C = torch.cat([emb_sketch, emb_photo], dim=0)
    S_text_2C = compute_similarity_matrix(emb_2C)
    design1_blocks = compute_block_stats(S_text_2C, K)
    
    # --- Design 2: Two separate C×C matrices ---
    print('[INFO] Building Design 2: Two Separate C×C Matrices...')
    S_text_sketch = compute_similarity_matrix(emb_sketch)
    S_text_photo = compute_similarity_matrix(emb_photo)
    design2_corr = 0.0
    if pearsonr:
        design2_corr, _ = pearsonr(flatten_upper_tri(S_text_sketch), flatten_upper_tri(S_text_photo))
    
    # --- Design 3: One shared C×C matrix ---
    print('[INFO] Building Design 3: One Shared C×C Matrix...')
    emb_shared = per_template[template_shared]
    S_text_shared = compute_similarity_matrix(emb_shared)
    
    # --- Save matrices and metrics ---
    matrices_for_viz = {
        'design1': S_text_2C,
        'design2_sketch': S_text_sketch,
        'design2_photo': S_text_photo,
        'design3': S_text_shared,
    }
    
    metrics = {
        'design1_blocks': design1_blocks,
        'design2_sketch_photo_correlation': design2_corr,
    }
    # 5. Save metrics
    # ---------------------------------
    metrics_path = out_dir / "initial_metrics.json"
    print(f"[INFO] Saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        # Convert numpy types to native python types before dumping
        native_metrics = convert_to_native_types(metrics)
        json.dump(native_metrics, f, indent=2)
        
    # --- Visualize ---
    print('[INFO] Generating visualizations...')
    visualize_initial_S_text(matrices_for_viz, design1_blocks, design2_corr, str(out_dir))
    
    print(f'\n[SUCCESS] Analysis complete. Results saved to: {args.out_dir}')
    print(f'  - Design 1 Block Stats: {design1_blocks}')
    print(f'  - Design 2 Sketch-Photo Correlation: {design2_corr:.4f}')

if __name__ == '__main__':
    main()
