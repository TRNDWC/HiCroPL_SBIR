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
        photo_feat_fixed, sketch_feat_fixed,
        text_zero_shot_photo, text_zero_shot_sketch,
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

    # --- L2: Visual consistency (sketch/photo vs augmented) ---
    if sketch_aug_feat is not None and photo_aug_feat is not None:
        loss_consistency = lambda_consistency * (
            cross_loss(sketch_feat, sketch_aug_feat, temperature) +
            cross_loss(photo_feat, photo_aug_feat, temperature)
        )
    else:
        loss_consistency = 0.0

    # --- L4: Cross-Entropy Loss (text - photo) + (text - sketch) ---
    loss_ce_photo = F.cross_entropy(logits_photo, label)
    loss_ce_sketch = F.cross_entropy(logits_sketch, label)
    loss_ce = lambda_ce * (loss_ce_photo + loss_ce_sketch)

    if getattr(args, 'enhance_text', False):
        # --- L3: Text Consistency (LLM-guided dual sketch/photo descriptions) ---
        loss_cons_text_sketch = 1.0 - F.cosine_similarity(text_feat_sketch, text_zero_shot_sketch, dim=-1)
        loss_cons_text_photo = 1.0 - F.cosine_similarity(text_feat_photo, text_zero_shot_photo, dim=-1)
        loss_cons_text = lambda_text_consistency * (loss_cons_text_sketch.mean() + loss_cons_text_photo.mean())
    
        # --- L_cons_visual_cross: Cross-anchor Visual to Text ---
        lambda_visual_cross = getattr(args, 'lambda_visual_cross', 0.1)
        text_distill_sketch_batch = text_distill_sketch[label]
        text_distill_photo_batch = text_distill_photo[label]
 
        loss_cons_visual_cross_sketch = 1.0 - F.cosine_similarity(sketch_feat, text_distill_sketch_batch, dim=-1)
        loss_cons_visual_cross_photo = 1.0 - F.cosine_similarity(photo_feat, text_distill_photo_batch, dim=-1)
        loss_cons_visual_cross = lambda_visual_cross * (loss_cons_visual_cross_sketch.mean() + loss_cons_visual_cross_photo.mean())
    else:
        loss_cons_text = 0.0
        loss_cons_visual_cross = 0.0

    # Total loss: L1 (cross-modal) + L2 (visual consistency) + L4 (CE) + L_cons_text + L_cons_visual_cross
    # Text distillation consistency remains disabled in this configuration.
    total_loss = loss_cross_modal + loss_consistency + loss_ce + loss_cons_text + loss_cons_visual_cross

    loss_dict = {
        'loss_cross_modal': loss_cross_modal,
        'loss_consistency': loss_consistency,
        'loss_ce': loss_ce,
        'loss_cons_text': loss_cons_text,
        'loss_cons_visual_cross': loss_cons_visual_cross
    }

    return total_loss, loss_dict
