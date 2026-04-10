import torch
import torch.nn as nn
import torch.nn.functional as F


def _min_max_normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    matrix_min = matrix.min()
    matrix_max = matrix.max()
    return (matrix - matrix_min) / (matrix_max - matrix_min + 1e-8)


def compute_structural_loss(prompted_features: torch.Tensor,
                            labels: torch.Tensor,
                            text_similarity_matrix: torch.Tensor) -> torch.Tensor:
    """Match batch feature geometry to frozen text prototype geometry."""
    if prompted_features.numel() == 0:
        return prompted_features.new_tensor(0.0)

    labels = labels.to(prompted_features.device).long()
    unique_classes = torch.unique(labels, sorted=True)
    if unique_classes.numel() < 2:
        return prompted_features.new_tensor(0.0)

    centroids = []
    for cls_idx in unique_classes:
        cls_mask = labels == cls_idx
        cls_centroid = prompted_features[cls_mask].mean(dim=0)
        centroids.append(F.normalize(cls_centroid, dim=0, eps=1e-8))

    centroids = torch.stack(centroids, dim=0)
    prompted_similarity = centroids @ centroids.t()

    target_similarity = text_similarity_matrix.index_select(0, unique_classes)
    target_similarity = target_similarity.index_select(1, unique_classes)

    prompted_similarity = _min_max_normalize_matrix(prompted_similarity)
    target_similarity = _min_max_normalize_matrix(target_similarity)

    k = unique_classes.numel()
    structural_loss = torch.norm(prompted_similarity - target_similarity, p='fro') ** 2
    structural_loss = structural_loss / (k * k)
    return structural_loss


def build_multimodal_text_similarity_matrix(photo_text_prototypes: torch.Tensor,
                                            sketch_text_prototypes: torch.Tensor) -> torch.Tensor:
    """Build a 2Kx2K target map from photo/sketch text prototypes."""
    photo_text = F.normalize(photo_text_prototypes, dim=-1, eps=1e-8)
    sketch_text = F.normalize(sketch_text_prototypes, dim=-1, eps=1e-8)

    photo_photo = photo_text @ photo_text.t()
    photo_sketch = photo_text @ sketch_text.t()
    sketch_photo = sketch_text @ photo_text.t()
    sketch_sketch = sketch_text @ sketch_text.t()

    top = torch.cat([photo_photo, photo_sketch], dim=1)
    bottom = torch.cat([sketch_photo, sketch_sketch], dim=1)
    return torch.cat([top, bottom], dim=0)


def compute_multimodal_structural_loss(photo_features: torch.Tensor,
                                       sketch_features: torch.Tensor,
                                       labels: torch.Tensor,
                                       multimodal_text_similarity_matrix: torch.Tensor) -> torch.Tensor:
    """Match 2Kx2K model geometry (photo+sketch) to 2Kx2K text-guided target map."""
    if photo_features.numel() == 0 or sketch_features.numel() == 0:
        return photo_features.new_tensor(0.0)

    labels = labels.to(photo_features.device).long()
    unique_classes = torch.unique(labels, sorted=True)
    if unique_classes.numel() < 2:
        return photo_features.new_tensor(0.0)

    photo_centroids = []
    sketch_centroids = []
    for cls_idx in unique_classes:
        cls_mask = labels == cls_idx
        photo_centroids.append(photo_features[cls_mask].mean(dim=0))
        sketch_centroids.append(sketch_features[cls_mask].mean(dim=0))

    photo_centroids = F.normalize(torch.stack(photo_centroids, dim=0), dim=1, eps=1e-8)
    sketch_centroids = F.normalize(torch.stack(sketch_centroids, dim=0), dim=1, eps=1e-8)

    model_nodes = torch.cat([photo_centroids, sketch_centroids], dim=0)
    model_similarity = model_nodes @ model_nodes.t()

    n_classes_total = multimodal_text_similarity_matrix.shape[0] // 2
    target_indices = torch.cat([
        unique_classes,
        unique_classes + n_classes_total,
    ], dim=0)
    target_similarity = multimodal_text_similarity_matrix.index_select(0, target_indices)
    target_similarity = target_similarity.index_select(1, target_indices)

    model_similarity = _min_max_normalize_matrix(model_similarity)
    target_similarity = _min_max_normalize_matrix(target_similarity)

    n = model_similarity.shape[0]
    structural_loss = torch.norm(model_similarity - target_similarity, p='fro') ** 2
    structural_loss = structural_loss / (n * n)
    return structural_loss


def build_text_similarity_matrix(text_prototypes: torch.Tensor) -> torch.Tensor:
    """Build a cosine similarity matrix from frozen text prototypes."""
    normalized = F.normalize(text_prototypes, dim=-1, eps=1e-8)
    return normalized @ normalized.t()

def cross_loss(feature_1, feature_2, temperature):
    device = feature_1.device
    labels = torch.cat([torch.arange(len(feature_1)) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    feature_1 = F.normalize(feature_1, dim=1)
    feature_2 = F.normalize(feature_2, dim=1)
    features = torch.cat((feature_1, feature_2), dim=0)  # (2*B, Feat_dim)

    similarity_matrix = torch.matmul(features, features.T)  # (2*B, 2*B)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # (2*B, 2*B - 1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # (2*B, 1)

    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # (2*B, 2*(B - 1))

    logits = torch.cat([positives, negatives], dim=1)
    labels_target = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature

    return F.cross_entropy(logits, labels_target)

def loss_fn_hicropl(args, features):
    """
    Combined Loss Function for HiCroPL-SBIR.
    
    Loss Components:
    L1: Triplet Loss (sketch, positive_photo, negative_photo) - Retrieval alignment
    L2: InfoNCE Loss (sketch - positive_photo) - Cross-modal alignment
    L3: InfoNCE Loss (sketch - sketch_aug) + (photo - photo_aug) - Consistency regularization
    L4: Cross-Entropy Loss (text - photo) + (text - sketch) - Classification
    L5: Cross-Entropy Loss (text - photo_aug) + (text - sketch_aug) - Augmented Classification
    """
    (
        photo_feat, logits_photo,
        sketch_feat, logits_sketch,
        neg_feat, label,
        photo_aug_feat, sketch_aug_feat,
        logits_photo_aug, logits_sketch_aug,
        text_feat_photo, text_feat_sketch,
        text_feat_fixed_photo, text_feat_fixed_sketch,
        photo_feat_fixed, sketch_feat_fixed,
    ) = features

    device = logits_photo.device
    label = label.to(device)
    
    # Get hyperparameters
    temperature = getattr(args, 'temperature', 0.07)
    lambda_triplet = getattr(args, 'lambda_triplet', 1.0)
    lambda_cross_modal = getattr(args, 'lambda_cross_modal', 1.0)
    lambda_consistency = getattr(args, 'lambda_consistency', 1.0)
    lambda_ce = getattr(args, 'lambda_ce', 1.0)
    lambda_ce_aug = getattr(args, 'lambda_ce_aug', 1.0)
    lambda_struct_photo = getattr(args, 'lambda_struct_photo', 0.1)
    lambda_struct_sketch = getattr(args, 'lambda_struct_sketch', 0.1)
    triplet_margin = getattr(args, 'triplet_margin', 0.3)

    # --- L1: Triplet Loss (sketch, positive_photo, negative_photo) ---
    # distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    # triplet = nn.TripletMarginWithDistanceLoss(
    #     distance_function=distance_fn,
    #     margin=triplet_margin
    # )
    # loss_triplet = lambda_triplet * triplet(sketch_feat, photo_feat, neg_feat)

    # --- L2: InfoNCE Loss (sketch - positive_photo) ---
    loss_cross_modal = lambda_cross_modal * cross_loss(sketch_feat, photo_feat, temperature)

    # --- L3: InfoNCE Loss (sketch - sketch_aug) + (photo - photo_aug) ---
    loss_consistency_sketch = cross_loss(sketch_feat, sketch_aug_feat, temperature)
    loss_consistency_photo = cross_loss(photo_feat, photo_aug_feat, temperature)
    loss_consistency = lambda_consistency * (loss_consistency_sketch + loss_consistency_photo)

    # --- L4: Cross-Entropy Loss (text - photo) + (text - sketch) ---
    loss_ce_photo = F.cross_entropy(logits_photo, label)
    loss_ce_sketch = F.cross_entropy(logits_sketch, label)
    loss_ce = lambda_ce * (loss_ce_photo + loss_ce_sketch)

    # --- L5: Multi-modal structural anchoring via 2Kx2K text target map ---
    text_similarity_multimodal = build_multimodal_text_similarity_matrix(
        photo_text_prototypes=text_feat_fixed_photo,
        sketch_text_prototypes=text_feat_fixed_sketch,
    )
    loss_struct_mm = compute_multimodal_structural_loss(
        photo_features=photo_feat,
        sketch_features=sketch_feat,
        labels=label,
        multimodal_text_similarity_matrix=text_similarity_multimodal,
    )
    lambda_struct_mm = 0.5 * (lambda_struct_photo + lambda_struct_sketch)
    loss_struct = lambda_struct_mm * loss_struct_mm

    # --- L6: Cross-Entropy Loss (text - photo_aug) + (text - sketch_aug) ---
    # loss_ce_photo_aug = F.cross_entropy(logits_photo_aug, label)
    # loss_ce_sketch_aug = F.cross_entropy(logits_sketch_aug, label)
    # loss_ce_aug = lambda_ce_aug * (loss_ce_photo_aug + loss_ce_sketch_aug)

    # Total loss
    total_loss = loss_cross_modal + loss_ce + loss_struct

    return total_loss
