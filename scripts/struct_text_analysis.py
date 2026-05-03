"""
Lightweight analysis for S_text vs visual geometry, designed for quick runs on Kaggle.
Produces: S_text (.pt/.csv), histograms, heatmap, S_text vs S_photo correlation and scatter,
clustering summary, and zero-shot neighbor probe.

Usage (example for Sketchy dataset):
python scripts/struct_text_analysis.py \
    --image-root /kaggle/input/.../Sketchy \
    --out-dir results/struct_text_analysis \
    --model ViT-B/32 \
    --templates "a photo of a {},a sketch of a {}" \
    --max-classes 100 --samples-per-class 5 --device cuda

The script expects an image folder structure: <image-root>/photo/<class_name>/*.jpg
                                            <image-root>/sketch/<class_name>/*.jpg

Key feature: Automatically samples from BOTH photo and sketch folders (if available) to compute
S_photo and S_sketch separately, then correlates each against S_text to validate the anchor quality
for both modalities.

Classname strings should match folder names (underscores allowed, converted to spaces).

Designed for speed: samples classes/images, batches encodings, template ensemble.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# local CLIP wrapper in repo
from src.clip import clip as _clip

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception:
    KMeans = None
    silhouette_score = None

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None


def clean_name(s):
    return s.replace('_', ' ')


def load_classnames(path):
    """Load from file, or auto-detect from folder if path is a directory."""
    if os.path.isdir(path):
        # Auto-detect from folder structure (photo/sketch style)
        p = Path(path)
        
        # Check for photo/sketch structure FIRST (most common for ZS-SBIR)
        photo_dir = p / 'photo'
        sketch_dir = p / 'sketch'
        
        print(f'[DEBUG] Checking for photo/sketch structure in: {path}')
        print(f'[DEBUG] photo/ exists: {photo_dir.exists()}, sketch/ exists: {sketch_dir.exists()}')
        
        if photo_dir.exists() and sketch_dir.exists():
            # Standard ZS-SBIR structure: get classes from photo/ folder
            candidates = sorted([c.name for c in photo_dir.iterdir() 
                               if c.is_dir() and not c.name.startswith('.')])
            print(f'[DEBUG] Found {len(candidates)} classes from photo/ subdirectory')
            if candidates:
                return candidates
        
        # Fallback: look for class folders directly under path
        print(f'[DEBUG] Trying direct subfolders (fallback)...')
        candidates = sorted([item.name for item in p.iterdir() 
                            if item.is_dir() and not item.name.startswith('.')])
        
        print(f'[DEBUG] Found direct folders: {len(candidates)} items: {candidates[:3]}...')
        
        # Filter out 'photo' and 'sketch' if they exist (not class folders)
        candidates = [c for c in candidates if c not in ['photo', 'sketch']]
        
        if candidates:
            print(f'Auto-detected {len(candidates)} classes from {path}')
            return sorted(candidates)
        raise RuntimeError(f'No classes found in {path}')
    else:
        # Load from file
        with open(path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines


def compute_text_embeddings(model, device, classnames, templates):
    # templates: list of format strings with a single {} placeholder
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for tmpl in templates:
            texts = [tmpl.format(clean_name(c)) for c in classnames]
            tokenized = _clip.tokenize(texts).to(device)
            emb = model.encode_text(tokenized)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeds.append(emb.cpu())
    # ensemble by mean (if multiple templates)
    embeds = torch.stack(all_embeds, dim=0).mean(dim=0)
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds


def compute_similarity_matrix(embeds):
    # embeds: [K, D] numpy or torch
    if isinstance(embeds, torch.Tensor):
        with torch.no_grad():
            sim = (embeds @ embeds.t()).cpu().numpy()
    else:
        sim = np.matmul(embeds, embeds.T)
    return sim


def sample_images_per_class(image_root, classnames, max_classes, samples_per_class, modality=None):
    """
    Sample images per class from image_root.
    If modality is provided (e.g. 'photo', 'sketch'), look for that subfolder.
    Otherwise try both photo/ and sketch/ subdirs.
    Returns list of (classname, [paths]).
    """
    image_root = Path(image_root)
    p = image_root
    
    # Determine which modality folder(s) to search
    if modality:
        # Explicit modality requested
        search_roots = [image_root / modality]
    else:
        # Try standard photo/sketch structure
        if (image_root / 'photo').exists() or (image_root / 'sketch').exists():
            search_roots = [image_root / 'photo', image_root / 'sketch']
        else:
            # Fall back to image_root directly
            search_roots = [image_root]
    
    print(f'  [DEBUG] Looking for images in {len(search_roots)} location(s):')
    for sr in search_roots:
        print(f'    - {sr} (exists: {sr.exists()})')
    
    available = []
    for search_root in search_roots:
        if not search_root.exists():
            print(f'  [DEBUG] Path does not exist: {search_root}')
            continue
            
        print(f'  [DEBUG] Scanning {search_root} for class folders...')
        for cname in classnames[:max_classes]:
            folder = search_root / cname
            if folder.exists() and folder.is_dir():
                files = list(folder.glob('*'))
                files = [p for p in files if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                if files:
                    available.append((cname, files))
        
        print(f'  [DEBUG] Found {len(available)} classes with image files')
    
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
    # sampled: list of (classname, [paths])
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
                except Exception:
                    continue
            if not imgs:
                continue
            imgs = torch.stack(imgs, dim=0).to(device)
            # encode in batches
            feats = []
            for i in range(0, len(imgs), batch_size):
                batch = imgs[i:i+batch_size]
                f = model.encode_image(batch)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu())
            feats = torch.cat(feats, dim=0)
            centroids[cname] = feats.mean(dim=0).numpy()
    return centroids


def flatten_upper_tri(mat):
    # return vector of upper-triangle (i<j)
    k = mat.shape[0]
    idx = np.triu_indices(k, k=1)
    return mat[idx]


def quick_stats(arr):
    return dict(mean=float(arr.mean()), std=float(arr.std()), min=float(arr.min()), max=float(arr.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classnames-file', default=None, help='Path to classnames.txt or auto-detect from image-root folder')
    parser.add_argument('--image-root', default=None, help='Root directory with class subdirectories. Auto-detects classnames if not provided.')
    parser.add_argument('--out-dir', default='results/struct_text_analysis')
    parser.add_argument('--model', default='ViT-B/32')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-classes', type=int, default=100)
    parser.add_argument('--samples-per-class', type=int, default=5)
    parser.add_argument('--templates', default='a photo of a {},a sketch of a {}', 
                        help='Comma-separated templates, e.g. "a photo of a {},a sketch of a {}"')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f'DEBUG: args.templates = "{args.templates}"')

    # Load classnames: from file, or auto-detect from image_root
    if args.classnames_file:
        classnames = load_classnames(args.classnames_file)
    elif args.image_root:
        classnames = load_classnames(args.image_root)
    else:
        raise RuntimeError('Must provide either --classnames-file or --image-root')
    
    K = len(classnames)
    print(f"Loaded {K} classnames, sampling up to {args.max_classes}")

    device = torch.device(args.device)
    print(f"Loading CLIP model {args.model} on {device}")
    clip_model, preprocess = _clip.load(args.model, device=device, jit=False)

    templates = [t.strip() for t in args.templates.split(',')]
    templates = [t for t in templates if '{}' in t]
    if not templates:
        raise RuntimeError(f'Provide at least one template with {{}} placeholder. Got: {args.templates}')
    print(f'Using {len(templates)} templates: {templates}')

    # Compute text embeddings (fast)
    print('Computing text embeddings (templates:', templates, ')')
    text_embeds = compute_text_embeddings(clip_model, device, classnames[:args.max_classes], templates)
    S_text = compute_similarity_matrix(text_embeds)
    torch.save(torch.from_numpy(S_text), os.path.join(args.out_dir, 'S_text.pt'))
    np.savetxt(os.path.join(args.out_dir, 'S_text.csv'), S_text, delimiter=',')

    # Diagnostics: histogram of off-diagonal
    offdiag = S_text[np.triu_indices(S_text.shape[0], k=1)]
    stats = quick_stats(offdiag)
    print('S_text off-diagonal stats:', stats)
    plt.figure(figsize=(6,4))
    plt.hist(offdiag, bins=60)
    plt.title('Histogram of S_text off-diagonal similarities')
    plt.savefig(os.path.join(args.out_dir, 'S_text_hist.png'))
    plt.close()

    # Heatmap (clipped)
    plt.figure(figsize=(6,6))
    plt.imshow(S_text, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('S_text heatmap')
    plt.savefig(os.path.join(args.out_dir, 'S_text_heatmap.png'))
    plt.close()

    # Optional clustering analysis
    if K <= args.max_classes and K >= 3 and KMeans is not None:
        try:
            k = min(10, max(2, K//10))
            X = text_embeds.numpy()
            km = KMeans(n_clusters=k, random_state=args.seed).fit(X)
            sil = silhouette_score(X, km.labels_)
            print(f'KMeans k={k}, silhouette={sil:.4f}')
            np.savetxt(os.path.join(args.out_dir, 'kmeans_labels.txt'), km.labels_.astype(int), fmt='%d')
        except Exception as e:
            print('Clustering failed:', e)

    # If image_root provided, compute visual centroids and compare
    if args.image_root:
        # Try to sample from both photo and sketch modalities
        for modality_name in ['photo', 'sketch']:
            print(f'\n--- Analyzing {modality_name.upper()} modality ---')
            sampled = sample_images_per_class(args.image_root, classnames[:args.max_classes], 
                                              args.max_classes, args.samples_per_class, 
                                              modality=modality_name)
            if not sampled:
                print(f'No {modality_name} images found; skipping')
                continue
            
            print(f'Sampled {len(sampled)} {modality_name} classes (up to {args.samples_per_class} images each)')
            centroids = compute_visual_class_centroids(clip_model, preprocess, device, sampled)
            
            # align order with classnames subset
            names = [c for c,_ in sampled if c in centroids]
            C = np.stack([centroids[n] for n in names], axis=0)
            S_visual = np.matmul(C, C.T)
            
            # save
            np.savetxt(os.path.join(args.out_dir, f'S_{modality_name}.csv'), S_visual, delimiter=',')
            torch.save(torch.from_numpy(S_visual), os.path.join(args.out_dir, f'S_{modality_name}.pt'))

            # compute correlation between S_text (restricted to sampled names) and S_visual
            idx_map = {name:i for i,name in enumerate(classnames[:args.max_classes])}
            indices = [idx_map[n] for n in names]
            S_text_sub = S_text[np.ix_(indices, indices)]
            v_text = flatten_upper_tri(S_text_sub)
            v_visual = flatten_upper_tri(S_visual)
            
            if pearsonr and spearmanr:
                pr, _ = pearsonr(v_text, v_visual)
                sr, _ = spearmanr(v_text, v_visual)
                print(f'Correlation S_text vs S_{modality_name}: Pearson={pr:.4f}, Spearman={sr:.4f}')
            else:
                # fallback numpy correlation if scipy not available
                pr = np.corrcoef(v_text, v_visual)[0, 1]
                print(f'Correlation S_text vs S_{modality_name}: Pearson={pr:.4f}')

            # Scatter plot
            plt.figure(figsize=(5,4))
            plt.scatter(v_text, v_visual, s=4, alpha=0.6)
            plt.xlabel('S_text (upper tri)')
            plt.ylabel(f'S_{modality_name} (upper tri)')
            if pearsonr and spearmanr:
                pr_title, _ = pearsonr(v_text, v_visual)
                sr_title, _ = spearmanr(v_text, v_visual)
                plt.title(f'S_text vs S_{modality_name}: Pearson={pr_title:.3f}, Spearman={sr_title:.3f}')
            else:
                pr_title = np.corrcoef(v_text, v_visual)[0, 1]
                plt.title(f'S_text vs S_{modality_name}: Pearson={pr_title:.3f}')
            plt.savefig(os.path.join(args.out_dir, f'S_text_vs_S_{modality_name}_scatter.png'))
            plt.close()

    # Zero-shot probe: if user provided unseen list file 'unseen.txt' in same dir as classnames-file
    unseen_path = None
    if args.classnames_file:
        unseen_path = os.path.join(os.path.dirname(args.classnames_file), 'unseen.txt')
    if unseen_path and os.path.exists(unseen_path):
        with open(unseen_path, 'r') as f:
            unseen = [l.strip() for l in f if l.strip()]
        if unseen:
            print(f'Found {len(unseen)} unseen classes; probing nearest seen neighbors')
            with torch.no_grad():
                tok = _clip.tokenize([t.format(clean_name(u)) for t in templates for u in unseen]).to(device)
                emb = clip_model.encode_text(tok)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                # if multiple templates, embeddings are consecutive blocks; average per unseen
                T = len(templates)
                emb = emb.view(T, len(unseen), -1).mean(dim=0)
                emb = emb.cpu().numpy()
            seen_emb = text_embeds.numpy()
            sims = emb @ seen_emb.T
            # for each unseen, list top-5 seen
            topk = 5
            neighbors = {u: [classnames[i] for i in sims[idx].argsort()[::-1][:topk]] for idx,u in enumerate(unseen)}
            for u, neigh in neighbors.items():
                print(f'unseen {u} -> neighbors: {neigh}')
            # save neighbors
            with open(os.path.join(args.out_dir, 'unseen_neighbors.txt'), 'w') as f:
                for u, neigh in neighbors.items():
                    f.write(u + '\t' + ','.join(neigh) + '\n')

    print('Done. Results saved to', args.out_dir)


if __name__ == '__main__':
    main()
