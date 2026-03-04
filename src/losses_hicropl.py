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
    
    Includes:
    1. Triplet Loss (Semantic retrieval alignment)
    2. Cross-Entropy Loss (Photo vs Text, Sketch vs Text)
    3. Consistency Loss (Photo vs Frozen Photo, Sketch vs Frozen Sketch)
    """
    (
        photo_feat, frozen_photo_feat, logits_photo,
        sketch_feat, frozen_sketch_feat, logits_sketch,
        neg_feat, label
    ) = features

    device = logits_photo.device
    label = label.to(device)

    # --- 1. Triplet Loss for Retrieval Alignment ---
    distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    triplet = nn.TripletMarginWithDistanceLoss(
        distance_function=distance_fn, margin=0.3
    )
    # Target: sketch should be closer to pos_photo than neg_photo
    loss_triplet = triplet(sketch_feat, photo_feat, neg_feat)

    # --- 2. Cross-Entropy Losses for Classification ---
    loss_ce_photo = F.cross_entropy(logits_photo, label)
    loss_ce_sk = F.cross_entropy(logits_sketch, label)
    
    # Weight classification loss based on arguments (default to 1.0 if not specified)
    lambda_ce = getattr(args, 'lambda_ce', 1.0)
    loss_cls = lambda_ce * (loss_ce_photo + loss_ce_sk)

    # --- 3. Consistency Losses ---
    # Prevents learned prompts from drifting too far from original frozen CLIP features
    loss_consist_photo = 1.0 - F.cosine_similarity(photo_feat, frozen_photo_feat).mean()
    loss_consist_sk = 1.0 - F.cosine_similarity(sketch_feat, frozen_sketch_feat).mean()
    
    # Weight consistency loss based on arguments (default to 0.1 if not specified)
    lambda_consist = getattr(args, 'lambda_consist', 1)
    loss_consist = lambda_consist * (loss_consist_photo + loss_consist_sk)

    # --- 4. InfoNCE Loss (sketch - positive_photo) ---
    temperature = getattr(args, 'temperature', 0.07)
    lambda_infonce = getattr(args, 'lambda_infonce', 1.0)
    loss_infonce = lambda_infonce * cross_loss(sketch_feat, photo_feat, temperature)

    # Total aggregate loss
    total_loss = loss_triplet + loss_cls + loss_consist + loss_infonce

    return total_loss
