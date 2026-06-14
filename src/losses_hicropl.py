import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_loss(feature_1, feature_2, temperature):
    """Symmetric NT-Xent / InfoNCE over two views.

    Treats (feature_1[i], feature_2[i]) as the only positive pair; every other sample in
    the 2*B stack is a negative.
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


def nt_xent(features_view1: torch.Tensor, features_view2: torch.Tensor, temperature: float = 0.07):
    """NT-Xent (SimCLR) between two feature views.

    Positive pair: (features_view1[i], features_view2[i]).
    All other 2B-2 pairs are negatives.
    """
    features_view1 = F.normalize(features_view1, dim=1)
    features_view2 = F.normalize(features_view2, dim=1)

    B = features_view1.shape[0]
    device = features_view1.device

    z = torch.cat([features_view1, features_view2], dim=0)   # (2B, D)
    logits = z @ z.t()                                        # (2B, 2B)
    mask = torch.eye(2 * B, dtype=torch.bool, device=device)
    logits = logits.masked_fill(mask, float('-inf'))
    logits = logits / temperature

    labels = torch.cat([
        torch.arange(B, 2 * B, device=device),
        torch.arange(0, B, device=device),
    ], dim=0).long()

    return F.cross_entropy(logits, labels)


def loss_fn_hicropl(args, features):
    """SBIR loss: CE + Triplet + NT-Xent (giống CoPrompt, bỏ Distillation).

    L_total = (CE_photo + CE_sketch) + Triplet(sketch, photo+, photo-) + NT-Xent(photo, sketch)
    """
    (
        photo_feat, logits_photo,
        sketch_feat, logits_sketch,
        neg_feat, label,
        *_
    ) = features

    device = logits_photo.device
    label = label.to(device)

    # Classification loss
    loss_ce = F.cross_entropy(logits_photo, label) + F.cross_entropy(logits_sketch, label)

    # Triplet loss
    triplet_margin = getattr(args, 'triplet_margin', 0.2)
    distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    triplet_fn = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn, margin=triplet_margin)
    loss_triplet = triplet_fn(sketch_feat, photo_feat, neg_feat)

    # NT-Xent cross-modal (photo ↔ sketch)
    loss_nt_xent = nt_xent(photo_feat, sketch_feat)

    return loss_ce  + loss_nt_xent
