import torch
import torch.nn as nn
import torch.nn.functional as F

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
    lambda_consist = getattr(args, 'lambda_consist', 0.1)
    loss_consist = lambda_consist * (loss_consist_photo + loss_consist_sk)

    # Total aggregate loss
    total_loss = loss_triplet + loss_cls + loss_consist

    return total_loss
