import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE

def load_data(results_dir, design_name):
    """Tải file CSV của một design cụ thể."""
    csv_path = results_dir / f"{design_name}.csv"
    if not csv_path.exists():
        print(f"Warning: Could not find {csv_path}. Skipping analysis for this file.")
        return None
    df = pd.read_csv(csv_path, index_col=0)
    # Loại bỏ prefix 'sketch: ', 'photo: ', 'shared: ' khỏi index/columns
    df.index = [i.split(': ')[-1] for i in df.index]
    df.columns = [c.split(': ')[-1] for c in df.columns]
    return df

def load_groups(groups_file):
    """Tải file JSON chứa thông tin nhóm lớp."""
    with open(groups_file, 'r') as f:
        groups = json.load(f)
    
    # Tạo một map từ class -> group để dễ tra cứu
    class_to_group = {}
    for group_name, classes in groups.items():
        for class_name in classes:
            class_to_group[class_name] = group_name
    return groups, class_to_group

def reorder_matrix(df, class_order):
    """Sắp xếp lại ma trận theo thứ tự lớp đã cho."""
    # Chỉ giữ lại các lớp có trong ma trận
    ordered_classes = [c for c in class_order if c in df.index]
    return df.loc[ordered_classes, ordered_classes]


def clustered_order(df):
    """Sắp xếp ma trận theo thứ tự lá từ hierarchical clustering."""
    dist = 1.0 - df.values
    dist = np.clip(dist, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    linkage = sch.linkage(condensed, method='average')
    leaves = sch.leaves_list(linkage)
    ordered_labels = df.index[leaves].tolist()
    return df.loc[ordered_labels, ordered_labels], linkage, ordered_labels

def calculate_intra_inter_ratio(df, group_map):
    """Tính toán tỷ lệ tương đồng nội nhóm / liên nhóm."""
    intra_group_sims = []
    inter_group_sims = []

    all_classes = df.index.tolist()
    
    for i in range(len(all_classes)):
        for j in range(i + 1, len(all_classes)): # Chỉ lấy tam giác trên
            class1 = all_classes[i]
            class2 = all_classes[j]
            
            group1 = group_map.get(class1)
            group2 = group_map.get(class2)
            
            if group1 is None or group2 is None:
                continue

            similarity = df.loc[class1, class2]
            
            if group1 == group2:
                intra_group_sims.append(similarity)
            else:
                inter_group_sims.append(similarity)

    if not intra_group_sims or not inter_group_sims:
        return 0.0

    avg_intra = np.mean(intra_group_sims)
    avg_inter = np.mean(inter_group_sims)
    
    return avg_intra / avg_inter if avg_inter != 0 else float('inf')


def visualize_heatmap(df, title, path, group_map=None, use_clustermap=False, linkage=None):
    """Vẽ heatmap đã được sắp xếp theo nhóm hoặc theo clustering tự động."""
    if use_clustermap:
        if linkage is not None:
            grid = sns.clustermap(
                df,
                cmap='viridis',
                figsize=(12, 12),
                row_linkage=linkage,
                col_linkage=linkage,
            )
        else:
            grid = sns.clustermap(df, cmap='viridis', figsize=(12, 12))
        grid.fig.suptitle(title, y=1.02, fontsize=16)
        grid.fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(grid.fig)
        return

    plt.figure(figsize=(12, 10))
    sns.heatmap(df, cmap='viridis', cbar=True)
    plt.title(title, fontsize=16)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_dendrogram(df, title, path):
    """Vẽ dendrogram phân cụm phân cấp."""
    # Chuyển đổi similarity matrix thành distance matrix
    dist_matrix = 1 - df.values
    dist_matrix = np.clip(dist_matrix, 0.0, None)
    np.fill_diagonal(dist_matrix, 0.0)
    condensed = squareform(dist_matrix, checks=False)

    # Thực hiện hierarchical clustering
    linked = sch.linkage(condensed, method='average')
    
    plt.figure(figsize=(20, 10))
    sch.dendrogram(linked, labels=df.index.tolist(), orientation='top', leaf_rotation=90)
    plt.title(title, fontsize=16)
    plt.ylabel("Distance")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_distribution(df, title, path):
    """Vẽ histogram và KDE của các giá trị similarity."""
    # Lấy các giá trị ở tam giác trên (không tính đường chéo)
    off_diagonal_values = df.values[np.triu_indices_from(df.values, k=1)]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(off_diagonal_values, kde=True, bins=50)
    plt.title(title, fontsize=16)
    plt.xlabel("Similarity")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_tsne(df, title, path, group_map=None):
    """Vẽ t-SNE projection."""
    tsne = TSNE(n_components=2, perplexity=min(30, len(df)-1), random_state=42, init='pca', learning_rate='auto')
    
    # t-SNE hoạt động tốt hơn với distance
    dist_matrix = 1 - df.values
    
    projections = tsne.fit_transform(dist_matrix)
    
    if group_map:
        # Lấy danh sách các nhóm duy nhất để tạo màu
        groups = sorted(list(set(group_map.values())))
        colors = plt.cm.get_cmap('tab20', len(groups))
        group_to_color = {group: colors(i) for i, group in enumerate(groups)}
    else:
        group_to_color = {'all': '#1f77b4'}
    
    plt.figure(figsize=(14, 10))
    
    # Vẽ các điểm
    for i, class_name in enumerate(df.index):
        group = group_map.get(class_name, 'all') if group_map else 'all'
        color = group_to_color.get(group, '#1f77b4')
        plt.scatter(projections[i, 0], projections[i, 1], color=color, label=group if i == 0 else "")
        plt.annotate(class_name, (projections[i, 0], projections[i, 1]), fontsize=8, alpha=0.7)

    # Tạo legend nếu có nhóm
    if group_map:
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=group,
                              markerfacecolor=color, markersize=10) for group, color in group_to_color.items()]
        plt.legend(handles=handles, title="Groups")

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Advanced analysis and visualization of S_text matrices.")
    parser.add_argument('--results-dir', required=True, help="Directory containing the initial CSV matrix files.")
    parser.add_argument('--groups-file', default=None, help="Optional JSON file defining class groups.")
    parser.add_argument('--out-dir', required=True, help="Directory to save advanced analysis results.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # --- 1. Load Data and Groups ---
    print("[INFO] Loading data and class groups...")
    groups = None
    class_to_group = {}
    class_order = None
    if args.groups_file:
        groups, class_to_group = load_groups(args.groups_file)
        # Tạo một thứ tự lớp nhất quán dựa trên nhóm
        class_order = [cls for group_classes in groups.values() for cls in sorted(group_classes)]
    else:
        print("[INFO] No groups file provided. Heatmaps will use automatic clustering order.")

    design_names = [
        'design2_S_text_sketch',
        'design2_S_text_photo',
        'design3_S_text_shared'
    ]
    
    all_metrics = {}

    # --- 2. Analyze each design ---
    for name in design_names:
        print(f"\n[INFO] Analyzing '{name}'...")
        df = load_data(results_dir, name)
        if df is None:
            continue

        if class_order is not None:
            # Sắp xếp lại ma trận theo nhóm
            df_reordered = reorder_matrix(df, class_order)
            use_clustermap = False
            heatmap_linkage = None
        else:
            # Sắp xếp lại ma trận theo cấu trúc nội tại
            df_reordered, heatmap_linkage, _ = clustered_order(df)
            use_clustermap = True
        
        # --- 3. Perform Visualizations ---
        print(f"  - Generating visualizations for {name}...")
        viz_out_dir = out_dir / name
        viz_out_dir.mkdir(exist_ok=True)

        # Heatmap
        visualize_heatmap(
            df_reordered,
            f"Sorted Heatmap - {name}" if class_order is not None else f"Clustered Heatmap - {name}",
            viz_out_dir / "1_sorted_heatmap.png",
            group_map=class_to_group if class_to_group else None,
            use_clustermap=use_clustermap,
            linkage=heatmap_linkage,
        )
        
        # Dendrogram
        visualize_dendrogram(df, f"Hierarchical Clustering - {name}", viz_out_dir / "2_dendrogram.png")
        
        # Distribution
        visualize_distribution(df, f"Similarity Distribution - {name}", viz_out_dir / "3_distribution.png")
        
        # t-SNE
        visualize_tsne(df, f"t-SNE Projection - {name}", viz_out_dir / "4_tsne.png", class_to_group)

        # --- 4. Calculate Quantitative Metrics ---
        print(f"  - Calculating quantitative metrics for {name}...")
        if class_to_group:
            ratio = calculate_intra_inter_ratio(df, class_to_group)
            all_metrics[name] = {
                'intra_inter_similarity_ratio': ratio
            }
            print(f"    - Intra/Inter Similarity Ratio: {ratio:.4f}")
        else:
            all_metrics[name] = {
                'intra_inter_similarity_ratio': None
            }
            print("    - Intra/Inter Similarity Ratio: skipped (no groups file)")

    # --- 5. Save Metrics ---
    metrics_path = out_dir / "advanced_metrics.json"
    print(f"\n[INFO] Saving all metrics to {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    print("\n[SUCCESS] Advanced analysis complete.")


if __name__ == '__main__':
    main()
