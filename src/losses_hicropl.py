import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize(x):
    return F.normalize(x, dim=-1)


def _normalize_matrix(m):
    """Min-max normalize a matrix to [0, 1] for stable Frobenius comparison."""
    min_v = m.min()
    max_v = m.max()
    return (m - min_v) / (max_v - min_v + 1e-8)


def compute_text_prototypes(classnames, clip_text_encoder, tokenize_fn, device,
                             ctx_prefix="a photo or a sketch of"):
    """
    Precompute frozen CLIP text prototype vectors for all seen classes.
    Call ONCE before training and store the result.

    Args:
        classnames        : list of seen class name strings
        clip_text_encoder : callable, e.g. clip_model_distill.encode_text
        tokenize_fn       : e.g. clip.tokenize
        device            : torch.device
        ctx_prefix        : prompt prefix (default "a photo or a sketch of")

    Returns:
        text_proto_mat : [n_cls, d] unit-norm prototype matrix, detached
    """
    text_prototypes = []
    with torch.no_grad():
        for cls in classnames:
            tokens = tokenize_fn(f"{ctx_prefix} {cls}").to(device)
            feat = clip_text_encoder(tokens)   # [1, d]
            feat = _normalize(feat)
            text_prototypes.append(feat)
    text_proto_mat = torch.cat(text_prototypes, dim=0)   # [n_cls, d]
    return text_proto_mat.detach()


# Keep old name as alias for backward compatibility
def compute_text_similarity_matrix(classnames, clip_text_encoder, tokenize_fn, device,
                                    ctx_prefix="a photo or a sketch of"):
    """Alias for compute_text_prototypes (returns [n_cls, d], not [n_cls, n_cls])."""
    return compute_text_prototypes(classnames, clip_text_encoder, tokenize_fn, device, ctx_prefix)


def _class_prototypes(features, labels):
    unique_labels = torch.unique(labels, sorted=True)
    prototypes = []
    prototype_labels = []

    for label in unique_labels:
        mask = labels == label
        if mask.any():
            prototype = _normalize(features[mask].mean(dim=0, keepdim=True))
            prototypes.append(prototype)
            prototype_labels.append(label)

    if not prototypes:
        return None, None

    return torch.cat(prototypes, dim=0), torch.stack(prototype_labels)


def semantic_prototype_loss(visual_features, semantic_features, labels):
    """
    Match class-level geometry between visual and text embeddings.

    Follows the reference exactly:
      1. Compute class centroids: mean in feature space -> normalize to unit sphere
      2. Build similarity matrices S_prompted [k,k] and S_text [k,k]
      3. Normalize both matrices to [0, 1] (min-max)
      4. Loss = ||S_prompted_norm - S_text_norm||^2_F / k^2
    """
    visual_proto, proto_labels = _class_prototypes(visual_features, labels)
    semantic_proto, semantic_labels = _class_prototypes(semantic_features, labels)

    if visual_proto is None or semantic_proto is None or visual_proto.shape[0] < 2:
        return visual_features.new_tensor(0.0)

    if not torch.equal(proto_labels, semantic_labels):
        raise ValueError("Visual and semantic prototypes must be aligned by the same class labels.")

    k = visual_proto.shape[0]

    # Step 2: similarity matrices
    S_prompted = visual_proto @ visual_proto.t()      # [k, k]
    S_text_mat = semantic_proto @ semantic_proto.t()  # [k, k]

    # Step 3: normalize both to [0, 1]
    S_prompted_norm = _normalize_matrix(S_prompted)
    S_text_norm     = _normalize_matrix(S_text_mat)

    # Step 4: Frobenius norm loss / k^2
    diff = S_prompted_norm - S_text_norm
    return torch.norm(diff, p='fro') ** 2 / (k * k)


def batch_structural_loss(prompted_features, semantic_features, labels):
    """
    Match class-level geometry between prompted visual embeddings and
    frozen text embeddings for the classes present in the current batch.

    semantic_features can be:
      - [B, d] : per-sample text features already aligned with batch
      - [N_cls, d]: full seen-class table (output of compute_text_similarity_matrix
                    row vectors); rows are indexed by integer label value
    """
    if semantic_features.shape[0] != labels.shape[0]:
        # Full table mode: index per-sample features from the class table
        semantic_features = semantic_features[labels]

    return semantic_prototype_loss(prompted_features, semantic_features, labels)

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

def loss_fn_hicropl(args, features, text_proto_matrix=None):
    """
    Combined Loss Function for HiCroPL-SBIR.

    Args:
        args             : config namespace
        features         : tuple returned by CustomCLIP.forward()
        text_proto_matrix: [n_seen, d] precomputed frozen text prototypes
                           (output of compute_text_prototypes). When provided,
                           this is used as the structural-loss geometry anchor
                           instead of the per-batch text_feat_fixed values.

    Loss Components:
    L2: InfoNCE Loss (sketch - positive_photo) - Cross-modal alignment
    L3: InfoNCE Loss (sketch - sketch_aug) + (photo - photo_aug) - Consistency
    L4: Cross-Entropy Loss (text - photo) + (text - sketch) - Classification
    L5: Semantic Prototype Anchoring
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
        *_
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

    # --- L5: Semantic Prototype Anchoring ---
    # Use precomputed text prototypes [n_seen, d] when available;
    # else fall back to per-batch fixed text features.
    if text_proto_matrix is not None:
        text_proto = text_proto_matrix.to(device)   # [n_seen, d]
        structural_loss_photo  = batch_structural_loss(photo_feat,  text_proto, label)
        structural_loss_sketch = batch_structural_loss(sketch_feat, text_proto, label)
    else:
        structural_loss_photo  = batch_structural_loss(photo_feat,  text_feat_fixed_photo,  label)
        structural_loss_sketch = batch_structural_loss(sketch_feat, text_feat_fixed_sketch, label)
    structural_loss = lambda_struct * (structural_loss_photo + structural_loss_sketch)

    # --- L5: Cross-Entropy Loss (text - photo_aug) + (text - sketch_aug) ---
    # loss_ce_photo_aug = F.cross_entropy(logits_photo_aug, label)
    # loss_ce_sketch_aug = F.cross_entropy(logits_sketch_aug, label)
    # loss_ce_aug = lambda_ce_aug * (loss_ce_photo_aug + loss_ce_sketch_aug)

    # Total loss
    total_loss = loss_cross_modal + loss_ce + structural_loss

    return total_loss
