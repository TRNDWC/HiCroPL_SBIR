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


def build_text_similarity_matrix(text_prototypes: torch.Tensor) -> torch.Tensor:
    """Build a cosine similarity matrix from frozen text prototypes."""
    normalized = F.normalize(text_prototypes, dim=-1, eps=1e-8)
    return normalized @ normalized.t()


def compute_text_similarity_matrix(classnames,
                                   clip_text_encoder,
                                   tokenize_fn,
                                   device,
                                   prompt_template: str) -> torch.Tensor:
    """Precompute a modality-specific text similarity matrix over seen classes."""
    text_prototypes = []
    with torch.no_grad():
        for cls in classnames:
            tokens = tokenize_fn(prompt_template.format(cls=cls)).to(device)
            text_feat = clip_text_encoder(tokens)
            text_prototypes.append(F.normalize(text_feat, dim=-1, eps=1e-8))

    text_prototypes = torch.cat(text_prototypes, dim=0)
    return build_text_similarity_matrix(text_prototypes).detach()

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

def loss_fn_hicropl(args,
                    features,
                    text_similarity_photo: torch.Tensor = None,
                    text_similarity_sketch: torch.Tensor = None):
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
    lambda_struct = getattr(args, 'lambda_struct', 0.1)
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

    # --- L5: Modality-specific structural anchoring ---
    lambda_struct_photo = getattr(args, 'lambda_struct_photo', None)
    lambda_struct_sketch = getattr(args, 'lambda_struct_sketch', None)
    if lambda_struct_photo is None:
        lambda_struct_photo = lambda_struct
    if lambda_struct_sketch is None:
        lambda_struct_sketch = lambda_struct

    if text_similarity_photo is None:
        text_similarity_photo = build_text_similarity_matrix(text_feat_fixed_photo)
    else:
        text_similarity_photo = text_similarity_photo.to(device)

    if text_similarity_sketch is None:
        text_similarity_sketch = build_text_similarity_matrix(text_feat_fixed_sketch)
    else:
        text_similarity_sketch = text_similarity_sketch.to(device)

    loss_struct_photo = compute_structural_loss(
        prompted_features=photo_feat,
        labels=label,
        text_similarity_matrix=text_similarity_photo,
    )
    loss_struct_sketch = compute_structural_loss(
        prompted_features=sketch_feat,
        labels=label,
        text_similarity_matrix=text_similarity_sketch,
    )
    loss_struct = lambda_struct_photo * loss_struct_photo + lambda_struct_sketch * loss_struct_sketch

    # --- L6: Cross-Entropy Loss (text - photo_aug) + (text - sketch_aug) ---
    # loss_ce_photo_aug = F.cross_entropy(logits_photo_aug, label)
    # loss_ce_sketch_aug = F.cross_entropy(logits_sketch_aug, label)
    # loss_ce_aug = lambda_ce_aug * (loss_ce_photo_aug + loss_ce_sketch_aug)

    # Total loss
    total_loss = loss_cross_modal + loss_struct

    return total_loss
