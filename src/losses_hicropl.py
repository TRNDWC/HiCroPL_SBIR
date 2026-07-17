import torch
import torch.nn as nn
import torch.nn.functional as F

def nt_xent_loss(features_view1, features_view2, temperature):
    """NT-Xent (SimCLR-style): pool of 2B = both modalities together, so
    same-modality pairs also act as negatives alongside cross-modal ones.
    """
    features_view1 = F.normalize(features_view1, dim=-1)
    features_view2 = F.normalize(features_view2, dim=-1)
    B = features_view1.shape[0]
    device = features_view1.device

    z = torch.cat([features_view1, features_view2], dim=0)

    logits = z @ z.t()  # (2B, 2B)
    mask = torch.eye(2 * B, dtype=torch.bool, device=device)
    logits = logits.masked_fill(mask, float('-inf'))
    logits = logits / temperature

    labels = torch.cat([
        torch.arange(B, 2 * B, device=device),
        torch.arange(0, B, device=device),
    ], dim=0).long()

    return F.cross_entropy(logits, labels)

def loss_fn_hicropl(args, features):
    """
    Combined Loss Function for HiCroPL-SBIR.

    Loss Components:
    L_cls: Cross-Entropy (text - photo) + (text - sketch) - Classification
    L_triplet: Sketch-photo-negative triplet (cosine distance)
    L_nt_xent: NT-Xent (photo + sketch pool) - cross-modal alignment
    """
    (
        photo_feat, logits_photo,
        sketch_feat, logits_sketch,
        neg_feat, label,
        text_feat_photo, text_feat_sketch,
    ) = features

    device = logits_photo.device
    label = label.to(device)

    # Get hyperparameters
    temperature = getattr(args, 'temperature', 0.07)
    lambda_cross_modal = getattr(args, 'lambda_cross_modal', 1.0)
    lambda_ce = getattr(args, 'lambda_ce', 1.0)
    triplet_margin = getattr(args, 'triplet_margin', 0.3)

    # --- L_cls: classification ---
    loss_ce_photo = F.cross_entropy(logits_photo, label)
    loss_ce_sketch = F.cross_entropy(logits_sketch, label)
    loss_cls = lambda_ce * (loss_ce_photo + loss_ce_sketch)

    # --- L_triplet: sketch anchor, photo positive, negative photo ---
    distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    triplet_fn = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn, margin=triplet_margin)
    loss_triplet = triplet_fn(sketch_feat, photo_feat, neg_feat)

    # --- L_nt_xent: cross-modal alignment ---
    loss_nt_xent = lambda_cross_modal * nt_xent_loss(photo_feat, sketch_feat, temperature)

    total_loss = loss_cls + loss_triplet + loss_nt_xent

    return total_loss
