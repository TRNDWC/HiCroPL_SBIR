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

def loss_fn_hicropl(args, features, return_components=False):
    """
    Combined Loss Function for HiCroPL-SBIR.

    Loss Components:
    L1: InfoNCE Loss (sketch - positive_photo) - Cross-modal alignment
    L4: Cross-Entropy Loss (text - photo) + (text - sketch) - Classification

    return_components is REPORTING ONLY: it changes the return type, never the
    arithmetic. The three terms handed back are the exact tensors that were
    summed (post-lambda), so logging them cannot drift from the loss that is
    actually backpropagated. Needed to read the aug term's share over training:
    under --aug_shared_encoder both views come from one encoder, which makes the
    InfoNCE term intrinsically easier, and a term that saturates near 0 is a
    different explanation of a result than a missing second encoder.
    """
    (
        photo_feat, logits_photo,
        sketch_feat, logits_sketch,
        neg_feat, label,
        text_feat_photo, text_feat_sketch,
        photo_aug_feat, sketch_aug_feat,
        text_desc_feat_photo, text_desc_feat_sketch,
    ) = features

    device = logits_photo.device
    label = label.to(device)

    # Get hyperparameters
    temperature = getattr(args, 'temperature', 0.07)
    lambda_cross_modal = getattr(args, 'lambda_cross_modal', 1.0)
    lambda_ce = getattr(args, 'lambda_ce', 1.0)
    lambda_aug = getattr(args, 'lambda_aug', 1.0)
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
    # photo_feat / sketch_feat come from the trainable prompted branch.
    # *_aug_feat come from clip_aug by default -- or, under --aug_shared_encoder
    # (Run A), from the SAME main encoder with the same prompts, in which case
    # this term becomes a plain siamese consistency loss on two views. Either
    # way the structure below is identical; only where the features came from
    # differs. clip_aug is frozen by freeze_all_but_bn --
    # its LayerNorm IS trainable and CustomCLIP.forward deliberately does not
    # wrap that encoder in no_grad, so gradient flows into BOTH arguments here.
    # cross_loss is symmetric (it concatenates the two views), so the aug
    # backbone's LayerNorm is trained by these terms rather than acting as a
    # fixed target.
    #
    # --lambda_aug scales BOTH terms together. It only scales: at 0.0 the branch
    # is still built and still runs two ViT forwards per step, and its params
    # stay in the optimizer with zero gradient. --disable_aug_branch is the
    # clean removal (features arrive as None, nothing is constructed).
    loss_aug = 0.0
    if photo_aug_feat is not None:
        loss_aug = lambda_aug * (cross_loss(photo_feat, photo_aug_feat, temperature)
                                 + cross_loss(sketch_feat, sketch_aug_feat, temperature))

    # --- L6: text template <-> VLM description, InfoNCE ---
    # Same construction as loss_aug on the image side: two views of the same
    # class through one encoder, pulled together. The template view carries the
    # learnable context; the description view carries VLM-written content and
    # rides the same context. Coefficient fixed at 1.0 -- no CLI flag was added
    # for it, matching how lambda_aug's structure looks at its default.
    loss_text = 0.0
    if text_desc_feat_photo is not None:
        loss_text = (cross_loss(text_feat_photo, text_desc_feat_photo, temperature)
                     + cross_loss(text_feat_sketch, text_desc_feat_sketch, temperature))

    total = loss_cross_modal + loss_ce + loss_aug + loss_text
    if return_components:
        return total, {
            'loss_cross_modal': loss_cross_modal,
            'loss_ce': loss_ce,
            'loss_aug': loss_aug,
            'loss_text': loss_text,
        }
    return total
