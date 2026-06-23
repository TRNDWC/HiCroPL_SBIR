import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_loss(feature_1, feature_2, temperature):
    """Symmetric NT-Xent / InfoNCE over two views (ported from feature/check).

    Treats (feature_1[i], feature_2[i]) as the only positive pair; every other sample in
    the 2*B stack is a negative. This is instance-level contrast (category labels ignored).
    """
    device = feature_1.device
    labels = torch.cat([torch.arange(len(feature_1)) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

    feature_1 = F.normalize(feature_1, dim=1)
    feature_2 = F.normalize(feature_2, dim=1)
    features = torch.cat((feature_1, feature_2), dim=0)            # (2B, D)

    similarity_matrix = torch.matmul(features, features.T)         # (2B, 2B)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)   # (2B, 1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1) / temperature
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
    return F.cross_entropy(logits, targets)


def loss_fn_hicropl(args, features):
    """SBIR loss: cross-modal alignment (NT-Xent InfoNCE or triplet) + text classification.

    L_total = lambda_cross_modal * L_align + lambda_ce * (L_cls_photo + L_cls_sketch)

    - L_align: symmetric NT-Xent between sketch and positive photo (default, in-batch
      negatives over the 2B stack), or cosine-distance triplet (sketch, photo+, photo-)
      when cross_modal_loss='triplet'.
    - L_cls: cross-entropy of cosine(visual, prompted-text) against class labels.
    """
    (
        photo_feat, logits_photo,
        sketch_feat, logits_sketch,
        neg_feat, label,
    ) = features

    device = logits_photo.device
    label = label.to(device)

    lambda_cross_modal = getattr(args, 'lambda_cross_modal', 1.0)
    lambda_ce = getattr(args, 'lambda_ce', 0.5)
    cross_modal_loss = getattr(args, 'cross_modal_loss', 'infonce')

    if cross_modal_loss == 'triplet':
        triplet_margin = getattr(args, 'triplet_margin', 0.3)
        distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        triplet_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=distance_fn, margin=triplet_margin,
        )
        loss_align = lambda_cross_modal * triplet_fn(sketch_feat, photo_feat, neg_feat)
    else:
        temperature = getattr(args, 'temperature', 0.07)
        loss_align = lambda_cross_modal * cross_loss(sketch_feat, photo_feat, temperature)

    loss_ce_photo = F.cross_entropy(logits_photo, label)
    loss_ce_sketch = F.cross_entropy(logits_sketch, label)
    loss_ce = lambda_ce * (loss_ce_photo + loss_ce_sketch)

    return loss_align + loss_ce
