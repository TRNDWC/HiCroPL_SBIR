import torch
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
    
    Loss terms:
    L1: InfoNCE(sketch, positive photo)
    L2: CE(photo/sketch adapted prompted feature, text adapted prompted feature)
        L3: CE(photo/sketch augmentation feature, text adapted prompted feature)
            L4: Consistency InfoNCE (visual only, separate branches)
                    InfoNCE(photo, photo_aug)
                      + InfoNCE(sketch, sketch_aug)
    """
    (
        photo_feat, logits_photo,
        sketch_feat, logits_sketch,
        _neg_feat, label,
        photo_aug_feat, sketch_aug_feat,
        logits_photo_aug, logits_sketch_aug,
        *_
    ) = features

    device = logits_photo.device
    label = label.to(device)
    
    # Get hyperparameters
    temperature = getattr(args, 'temperature', 0.07)
    lambda_cross_modal = getattr(args, 'lambda_cross_modal', 1.0)
    lambda_consistency = getattr(args, 'lambda_consistency', 1.0)
    lambda_ce = getattr(args, 'lambda_ce', 1.0)
    lambda_ce_aug = getattr(args, 'lambda_ce_aug', 0.0)

    # --- L1: InfoNCE(sketch, positive photo) ---
    loss_cross_modal = 1 * cross_loss(sketch_feat, photo_feat, temperature)

    # --- L2: CE(main adapted prompted image features, adapted prompted text features) ---
    loss_ce_photo = F.cross_entropy(logits_photo, label)
    loss_ce_sketch = F.cross_entropy(logits_sketch, label)
    loss_ce = 1 * (loss_ce_photo + loss_ce_sketch)

    # --- L3: CE(augmentation image features, adapted prompted text features) ---
    loss_ce_photo_aug = F.cross_entropy(logits_photo_aug, label)
    loss_ce_sketch_aug = F.cross_entropy(logits_sketch_aug, label)
    loss_ce_aug = 1 * (loss_ce_photo_aug + loss_ce_sketch_aug)

    # --- L4: Consistency InfoNCE (visual only, separate photo/sketch) ---
    loss_consistency_photo = cross_loss(photo_feat, photo_aug_feat, temperature)
    loss_consistency_sketch = cross_loss(sketch_feat, sketch_aug_feat, temperature)
    loss_consistency = lambda_consistency * (loss_consistency_photo + loss_consistency_sketch)

    # Total loss = L1 + L2 + L3 + L4.
    total_loss = (
        loss_cross_modal
        + loss_ce
        + loss_ce_aug
        + loss_consistency
    )

    return total_loss
