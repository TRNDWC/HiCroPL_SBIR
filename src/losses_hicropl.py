import torch
import torch.nn as nn
import torch.nn.functional as F

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
    L1: InfoNCE Loss (sketch - positive_photo) - Cross-modal alignment
    L4: Cross-Entropy Loss (text - photo) + (text - sketch) - Classification
    """
    (
        photo_feat, logits_photo,
        sketch_feat, logits_sketch,
        neg_feat, label,
        text_feat_photo, text_feat_sketch,
        photo_aug_feat, sketch_aug_feat,
    ) = features

    device = logits_photo.device
    label = label.to(device)

    # Get hyperparameters
    temperature = getattr(args, 'temperature', 0.07)
    lambda_cross_modal = getattr(args, 'lambda_cross_modal', 1.0)
    lambda_ce = getattr(args, 'lambda_ce', 1.0)
    triplet_margin = getattr(args, 'triplet_margin', 0.3)
    use_triplet_l1 = getattr(args, 'eval_mode', 'category') == 'fine_grained' or getattr(args, 'use_triplet_l1', False)

    # --- L1: cross-modal alignment ---
    # Category mode keeps the original InfoNCE objective.
    # Fine-grained mode can replace it with triplet loss using the paired negative photo.
    if use_triplet_l1:
        dist_pos = 1.0 - F.cosine_similarity(sketch_feat, photo_feat)
        dist_neg = 1.0 - F.cosine_similarity(sketch_feat, neg_feat)
        loss_cross_modal = lambda_cross_modal * F.relu(dist_pos - dist_neg + triplet_margin).mean()
    else:
        loss_cross_modal = lambda_cross_modal * cross_loss(sketch_feat, photo_feat, temperature)

    # --- L4: Cross-Entropy Loss (text - photo) + (text - sketch) ---
    loss_ce_photo = F.cross_entropy(logits_photo, label)
    loss_ce_sketch = F.cross_entropy(logits_sketch, label)
    loss_ce = lambda_ce * (loss_ce_photo + loss_ce_sketch)

    # --- L5: augmentation branch, InfoNCE(view goc, view augmented) ---
    # photo_feat / sketch_feat come from the trainable prompted branch;
    # *_aug_feat come from the frozen vanilla CLIP and carry no gradient. All
    # gradient therefore flows through the first argument only -- the frozen
    # features act as fixed positives. Weight is a fixed 1.0 by design: there
    # is no lambda flag, only --disable_aug_branch, which removes the branch
    # entirely (both features arrive as None).
    loss_aug = 0.0
    if photo_aug_feat is not None:
        loss_aug = (cross_loss(photo_feat, photo_aug_feat, temperature)
                    + cross_loss(sketch_feat, sketch_aug_feat, temperature))

    return loss_cross_modal + loss_ce + loss_aug
