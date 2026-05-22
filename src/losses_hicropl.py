import torch.nn as nn
import torch.nn.functional as F


def loss_fn_hicropl(args, features):
    """CLIP-AT loss (Sain et al. CVPR'23, Eq. 2 + Eq. 3).

    L_total = L_Tri + lambda_1 * (L_pcls + L_scls)

    L_Tri:  max{0, mu + d(f_s, f_p) - d(f_s, f_n)} with d(a, b) = 1 - cos(a, b)
    L_*cls: cross-entropy of (visual @ text_template.T) * logit_scale vs labels

    Paper hyperparameters: mu = 0.3, lambda_1 = 0.5.
    """
    (
        photo_feat, logits_photo,
        sketch_feat, logits_sketch,
        neg_feat, label,
        *_,
    ) = features

    device = logits_photo.device
    label = label.to(device)

    lambda_cross_modal = getattr(args, 'lambda_cross_modal', 1.0)
    lambda_ce = getattr(args, 'lambda_ce', 0.5)
    triplet_margin = getattr(args, 'triplet_margin', 0.3)

    distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    triplet_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=distance_fn, margin=triplet_margin,
    )

    loss_triplet = lambda_cross_modal * triplet_fn(sketch_feat, photo_feat, neg_feat)

    loss_ce_photo = F.cross_entropy(logits_photo, label)
    loss_ce_sketch = F.cross_entropy(logits_sketch, label)
    loss_ce = lambda_ce * (loss_ce_photo + loss_ce_sketch)

    return loss_triplet + loss_ce
