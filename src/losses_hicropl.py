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

def attribute_losses(v, label, attr_emb, neg_emb, logit_scale):
    """ArGue-inspired attribute-guided auxiliary losses (adaptation, not a
    reimplementation of Tian et al., CVPR 2024 -- see report).

    v         : (B, D) L2-normalized feature (sketch_feat or photo_feat,
                selected by --attr_branch upstream)
    label     : (B,) int64 class index, same indexing space as logits_photo/
                logits_sketch (i.e. same order as `classnames`)
    attr_emb  : A, (n_cls, M, D) frozen L2-normalized attribute embeddings,
                row order EXACTLY matching `classnames` (enforced at load
                time in CustomCLIP.__init__, see model_hicropl.py)
    neg_emb   : N, (K, D) frozen L2-normalized non-discriminative attribute
                embeddings (see tools/gen_neg_bank.py)
    logit_scale : s, the model's own (frozen) logit_scale.exp() -- NOT a
                hard-coded constant, per project constraint #2.

    Returns (L_attr, L_neg), both frozen-attribute-only (A and N are
    .detach()'d here as a second, redundant safety net on top of already
    being non-persistent, non-trainable buffers -- no gradient can reach A
    or N through either loss).
    """
    A = attr_emb.detach()
    N = neg_emb.detach()

    # L_attr: auxiliary classifier built entirely from frozen attribute text.
    sim = torch.einsum('bd,cmd->bcm', v, A)         # (B, n_cls, M)
    logits_attr = logit_scale * sim.mean(dim=2)      # (B, n_cls)
    L_attr = F.cross_entropy(logits_attr, label)

    # L_neg: push the distribution over non-discriminative ("negative")
    # attributes toward uniform (maximize entropy) -- discourages the
    # branch from leaning on attribute content that carries no
    # class-discriminative signal in the first place.
    p_neg = F.softmax(logit_scale * (v @ N.t()), dim=1)  # (B, K)
    H = -(p_neg * torch.log(p_neg + 1e-8)).sum(1).mean()
    L_neg = -H

    return L_attr, L_neg


def loss_fn_hicropl(args, features, model=None):
    """
    Combined Loss Function for HiCroPL-SBIR.

    Loss Components:
    L1: InfoNCE Loss (sketch - positive_photo) - Cross-modal alignment
    L4: Cross-Entropy Loss (text - photo) + (text - sketch) - Classification
    L_attr, L_neg: optional attribute-guided auxiliary losses (--use_attr_loss),
        see attribute_losses() above. Applied to exactly ONE branch
        (--attr_branch, default 'sketch'), never both at once.
    """
    (
        photo_feat, logits_photo,
        sketch_feat, logits_sketch,
        neg_feat, label,
        text_feat_photo, text_feat_sketch,
        logit_scale,
    ) = features

    device = logits_photo.device
    label = label.to(device)

    # Get hyperparameters
    temperature = getattr(args, 'temperature', 0.07)
    lambda_cross_modal = getattr(args, 'lambda_cross_modal', 1.0)
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

    # --- L4: Cross-Entropy Loss (text - photo) + (text - sketch) ---
    loss_ce_photo = F.cross_entropy(logits_photo, label)
    loss_ce_sketch = F.cross_entropy(logits_sketch, label)
    loss_ce = lambda_ce * (loss_ce_photo + loss_ce_sketch)

    total = loss_cross_modal + loss_ce

    use_attr_loss = getattr(args, 'use_attr_loss', False)
    if use_attr_loss:
        if model is None or not hasattr(model, 'attr_emb') or not hasattr(model, 'neg_emb'):
            raise RuntimeError(
                "--use_attr_loss is set but loss_fn_hicropl() was not given a `model` "
                "with attr_emb/neg_emb buffers -- CustomCLIP.__init__ only registers "
                "them when cfg.use_attr_loss=True; check the two are in sync."
            )
        attr_branch = getattr(args, 'attr_branch', 'sketch')
        if attr_branch == 'sketch':
            v = sketch_feat
        elif attr_branch == 'photo':
            v = photo_feat
        else:
            raise ValueError(f"--attr_branch must be 'sketch' or 'photo', got {attr_branch!r}")

        lambda_attr = getattr(args, 'lambda_attr', 1.0)
        lambda_neg = getattr(args, 'lambda_neg', 0.5)
        L_attr, L_neg = attribute_losses(v, label, model.attr_emb, model.neg_emb, logit_scale)
        total = total + lambda_attr * L_attr + lambda_neg * L_neg

    return total
