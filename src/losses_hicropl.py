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

def coral_loss(source, target):
    """Deep CORAL (Sun & Saenko, ECCV-W 2016): align 2nd-order statistics
    (feature covariance) between two domain batches.

        L_CORAL = 1 / (4 d^2) * || C_source - C_target ||_F^2

    source, target: (B, D) each -- does not require paired/same-size batches
    beyond matching D. Meant to close the CLIP "cone effect" / modality gap
    between photo and sketch batch distributions, upstream of whatever
    cross-domain prompt exchange consumes them.
    """
    d = source.shape[1]
    source_c = source - source.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    cov_source = source_c.t() @ source_c / (source.shape[0] - 1)
    cov_target = target_c.t() @ target_c / (target.shape[0] - 1)
    return (cov_source - cov_target).pow(2).sum() / (4 * d * d)


def loss_fn_hicropl(args, features):
    """
    Combined Loss Function for HiCroPL-SBIR.

    Loss Components:
    L_cls: Cross-Entropy (text - photo) + (text - sketch) - Classification
    L_nt_xent: NT-Xent (photo + sketch pool) - cross-modal alignment
    L_coral: Deep CORAL - align photo/sketch batch covariance (domain gap)
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
    lambda_coral = getattr(args, 'lambda_coral', 0.0)

    # --- L_cls: classification ---
    loss_ce_photo = F.cross_entropy(logits_photo, label)
    loss_ce_sketch = F.cross_entropy(logits_sketch, label)
    loss_cls = lambda_ce * (loss_ce_photo + loss_ce_sketch)

    # --- L_nt_xent: cross-modal alignment ---
    loss_nt_xent = lambda_cross_modal * nt_xent_loss(photo_feat, sketch_feat, temperature)

    total_loss = loss_cls + loss_nt_xent

    # --- L_coral: close the photo/sketch domain gap (off by default) ---
    if lambda_coral > 0:
        total_loss = total_loss + lambda_coral * coral_loss(photo_feat, sketch_feat)

    return total_loss
