import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce(sketch_feat, photo_feat, temperature):
    """Base InfoNCE (sketch -> photo).

    Query = sketch, keys = in-batch photos. For sketch i the positive is photo i (diagonal);
    the other B-1 photos are negatives. Single direction, matching the SBIR retrieval
    objective (query a photo gallery with a sketch).
    """
    sketch = F.normalize(sketch_feat, dim=1)
    photo = F.normalize(photo_feat, dim=1)
    logits = sketch @ photo.t() / temperature          # (B, B)
    labels = torch.arange(sketch.shape[0], device=sketch.device)
    return F.cross_entropy(logits, labels)


def loss_fn_hicropl(args, features):
    """SBIR loss: cross-modal alignment (InfoNCE or triplet) + text classification.

    L_total = lambda_cross_modal * L_align + lambda_ce * (L_cls_photo + L_cls_sketch)

    - L_align: InfoNCE between sketch and positive photo (default, in-batch negatives),
      or cosine-distance triplet (sketch, photo+, photo-) when cross_modal_loss='triplet'.
    - L_cls: cross-entropy of cosine(visual, prompted-text) against class labels.
    """
    (
        photo_feat, logits_photo,
        sketch_feat, logits_sketch,
        neg_feat, label,
        text_feat_photo, text_feat_sketch,
        *_
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
        loss_align = lambda_cross_modal * info_nce(sketch_feat, photo_feat, temperature)

    loss_ce_photo = F.cross_entropy(logits_photo, label)
    loss_ce_sketch = F.cross_entropy(logits_sketch, label)
    loss_ce = lambda_ce * (loss_ce_photo + loss_ce_sketch)

    return loss_align + loss_ce
