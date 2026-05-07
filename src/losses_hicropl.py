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
    L2: InfoNCE Loss (sketch - sketch_aug) + (photo - photo_aug) - Visual consistency regularization
    L3: InfoNCE Loss (text_sketch - text_sketch_distill) + (text_photo - text_photo_distill) - Text consistency regularization
    L4: Cross-Entropy Loss (text - photo) + (text - sketch) - Classification
    """
    (
        photo_feat, logits_photo,
        sketch_feat, logits_sketch,
        neg_feat, label,
        photo_aug_feat, sketch_aug_feat,
        logits_photo_aug, logits_sketch_aug,
        text_feat_photo, text_feat_sketch,
        text_distill_photo, text_distill_sketch,
        *_
    ) = features

    device = logits_photo.device
    label = label.to(device)
    
    # Get hyperparameters
    temperature = getattr(args, 'temperature', 0.07)
    lambda_cross_modal = getattr(args, 'lambda_cross_modal', 1.0)
    lambda_consistency = getattr(args, 'lambda_consistency', 1.0)
    lambda_text_consistency = getattr(args, 'lambda_text_consistency', lambda_consistency)
    lambda_ce = getattr(args, 'lambda_ce', 1.0)

    # --- L1: InfoNCE Loss (sketch - positive_photo) ---
    loss_cross_modal = lambda_cross_modal * cross_loss(sketch_feat, photo_feat, temperature)

    # --- L2: Visual distill consistency ---
    loss_consistency_sketch = cross_loss(sketch_feat, sketch_aug_feat, temperature)
    loss_consistency_photo = cross_loss(photo_feat, photo_aug_feat, temperature)
    loss_consistency = lambda_consistency * (
        loss_consistency_sketch
        + loss_consistency_photo
    )

    # --- L3: Text distill consistency ---
    loss_text_consistency_sketch = cross_loss(text_feat_sketch, text_distill_sketch, temperature)
    loss_text_consistency_photo = cross_loss(text_feat_photo, text_distill_photo, temperature)
    loss_text_consistency = lambda_text_consistency * (
        loss_text_consistency_sketch
        + loss_text_consistency_photo
    )

    # --- L4: Cross-Entropy Loss (text - photo) + (text - sketch) ---
    loss_ce_photo = F.cross_entropy(logits_photo, label)
    loss_ce_sketch = F.cross_entropy(logits_sketch, label)
    loss_ce = lambda_ce * (loss_ce_photo + loss_ce_sketch)

    # Total loss
    total_loss = loss_cross_modal + loss_ce + loss_consistency + loss_text_consistency

    return total_loss
