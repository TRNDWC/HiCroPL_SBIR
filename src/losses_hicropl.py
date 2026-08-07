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

def supcon_loss(feature_1, feature_2, labels, temperature):
    """Supervised contrastive — MỌI mẫu cùng lớp là positive.

    `cross_loss` (NT-Xent) chỉ coi đúng một cặp ghép là positive; mọi ảnh khác
    trong batch là negative, KỂ CẢ ảnh cùng lớp. Với batch 128 và 104 lớp, mỗi
    hàng có ~2.4 false negative — chỉ ~1% về số lượng, nhưng chúng là những
    negative GIỐNG NHẤT nên chi phối gradient của InfoNCE.
    """
    device = feature_1.device
    f = torch.cat([F.normalize(feature_1, dim=1), F.normalize(feature_2, dim=1)], dim=0)
    lab = torch.cat([labels, labels], dim=0).view(-1, 1)

    pos = (lab == lab.t()).float().to(device)
    eye = torch.eye(len(f), dtype=torch.bool, device=device)
    pos = pos.masked_fill(eye, 0)                      # bỏ chính nó khỏi positive

    logits = (f @ f.t()) / temperature
    logits = logits.masked_fill(eye, -1e9)             # và khỏi mẫu số
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    n_pos = pos.sum(1)
    valid = n_pos > 0                                  # hàng không có positive thì bỏ
    if not valid.any():
        return torch.zeros((), device=device)
    return -((pos * log_prob).sum(1)[valid] / n_pos[valid]).mean()


def text_align_loss(feat, target, mode):
    """Căn chỉnh `feat` với `target` (đã chuẩn hoá) theo một trong ba dạng.

    'legacy' — dạng gốc `1 - cos(a+b, a)`. Với vector đã chuẩn hoá nó bằng
        `1 - sqrt((1 + a·b)/2)`, tức chỉ là một reparameterization đơn điệu của
        cosine với gradient bão hoà. Không sai nhưng không giải thích được.
    'direct' — `1 - cos(a, b)`. Cùng ý đồ, viết thẳng.
    'rel'    — KL giữa phân bố tương đồng của `feat` và của `target` TRÊN CÁC LỚP.
        Chỉ so sánh THỨ HẠNG giữa các lớp nên tôn trọng modality gap của CLIP:
        đặc trưng ảnh và text nằm trong hai nón tách rời, cos(image, text) chỉ
        ~0.2-0.3 ngay cả với cặp khớp hoàn hảo, nên ép căn chỉnh TUYỆT ĐỐI sẽ
        đẩy đặc trưng ảnh ra khỏi manifold của CLIP.
    """
    if mode == 'legacy':
        return (1.0 - F.cosine_similarity(feat + target, feat, dim=-1)).mean()
    if mode == 'direct':
        return (1.0 - F.cosine_similarity(feat, target, dim=-1)).mean()
    raise ValueError(f'text_align_mode không hợp lệ: {mode}')


def relational_text_loss(feat, feat_frozen, protos, temperature=0.07):
    """KL( softmax(feat·Pᵀ/τ) ‖ softmax(feat_frozen·Pᵀ/τ) ) — mục tiêu QUAN HỆ.

    CÙNG bộ prototype text `P`, KHÁC đặc trưng: nhánh prompted phải xếp hạng các
    lớp giống như nhánh frozen. Thay vì kéo đặc trưng ảnh về đặc trưng text (đi
    ngược modality gap), chỉ ràng buộc THỨ HẠNG — đặc trưng được tự do dịch
    chuyển miễn giữ nguyên cấu trúc tương đối giữa các lớp.

    `feat_frozen` là mục tiêu cố định nên detach; nếu không, cách rẻ nhất để
    giảm loss là kéo mục tiêu về phía nguồn, đúng lỗi đã gặp ở nhánh distill.
    """
    p = F.log_softmax(feat @ protos.t() / temperature, dim=-1)
    q = F.softmax(feat_frozen.detach() @ protos.t() / temperature, dim=-1)
    return F.kl_div(p, q, reduction='batchmean')


def uses_triplet(args):
    """Điều kiện DUY NHẤT quyết định `neg_feat` có được dùng hay không.

    Đặt ở đây để `CustomCLIP.forward` (nơi quyết định có chạy encoder cho ảnh
    negative hay không) và `loss_fn_hicropl` không thể lệch nhau. Nếu hai nơi
    tự viết lại điều kiện, một ngày nào đó chúng sẽ khác nhau và loss sẽ nhận
    `neg_feat=None` mà không có lỗi rõ ràng.
    """
    return (getattr(args, 'eval_mode', 'category') == 'fine_grained'
            or getattr(args, 'use_triplet_l1', False))


def loss_fn_hicropl(args, features):
    """
    Combined Loss Function for HiCroPL-SBIR.
    
    Loss Components:
    L1: InfoNCE Loss (sketch - positive_photo) - Cross-modal alignment
    L2: InfoNCE Loss (sketch - sketch_aug) + (photo - photo_aug) - Visual consistency regularization
    L3: InfoNCE Loss (text_sketch - text_sketch_distill) + (text_photo - text_photo_distill) - Text consistency regularization
    L4: Cross-Entropy Loss (text - photo) + (text - sketch) - Classification
    """
    (
        photo_feat, logits_photo,
        sketch_feat, logits_sketch,
        neg_feat, label,
        photo_aug_feat, sketch_aug_feat,
        logits_photo_aug, logits_sketch_aug,
        text_feat_photo, text_feat_sketch,
        text_distill_photo, text_distill_sketch,
        photo_feat_fixed, sketch_feat_fixed,
        *_
    ) = features

    device = logits_photo.device
    label = label.to(device)
    
    # Get hyperparameters
    temperature = getattr(args, 'temperature', 0.07)
    lambda_cross_modal = getattr(args, 'lambda_cross_modal', 1.0)
    lambda_consistency = getattr(args, 'lambda_consistency', 1.0)
    lambda_text_consistency = getattr(args, 'lambda_text_consistency', lambda_consistency)
    lambda_ce = getattr(args, 'lambda_ce', 1.0)
    triplet_margin = getattr(args, 'triplet_margin', 0.3)
    use_triplet_l1 = uses_triplet(args)

    # --- L1: cross-modal alignment ---
    # Category mode keeps the original InfoNCE objective.
    # Fine-grained mode can replace it with triplet loss using the paired negative photo.
    if use_triplet_l1:
        dist_pos = 1.0 - F.cosine_similarity(sketch_feat, photo_feat)
        dist_neg = 1.0 - F.cosine_similarity(sketch_feat, neg_feat)
        loss_cross_modal = lambda_cross_modal * F.relu(dist_pos - dist_neg + triplet_margin).mean()
    elif getattr(args, 'supcon', False):
        loss_cross_modal = lambda_cross_modal * supcon_loss(
            sketch_feat, photo_feat, label, temperature)
    else:
        loss_cross_modal = lambda_cross_modal * cross_loss(sketch_feat, photo_feat, temperature)

    # --- L2: Visual consistency (sketch/photo vs augmented) ---
    if sketch_aug_feat is not None and photo_aug_feat is not None:
        loss_consistency = lambda_consistency * (
            cross_loss(sketch_feat, sketch_aug_feat, temperature) +
            cross_loss(photo_feat, photo_aug_feat, temperature)
        )
    else:
        loss_consistency = 0.0

    # --- L4: Cross-Entropy Loss (text - photo) + (text - sketch) ---
    loss_ce_photo = F.cross_entropy(logits_photo, label)
    loss_ce_sketch = F.cross_entropy(logits_sketch, label)
    loss_ce = lambda_ce * (loss_ce_photo + loss_ce_sketch)

    if getattr(args, 'enhance_text', False):
        mode = getattr(args, 'text_align_mode', 'legacy')
        lambda_visual_cross = getattr(args, 'lambda_visual_cross', 0.1)

        # --- L3: Text Consistency (LLM-guided dual sketch/photo descriptions) ---
        # Cả hai vế đều là đặc trưng TEXT nên không dính modality gap; chỉ cần
        # bỏ cách viết vòng vo của bản gốc.
        loss_cons_text = lambda_text_consistency * (
            text_align_loss(text_feat_sketch, text_distill_sketch,
                            'direct' if mode != 'legacy' else 'legacy')
            + text_align_loss(text_feat_photo, text_distill_photo,
                              'direct' if mode != 'legacy' else 'legacy'))

        # --- L_cons_visual_cross: Cross-anchor Visual to Text ---
        # Đây MỚI là chỗ modality gap gây hại: kéo đặc trưng ẢNH về đặc trưng
        # TEXT. Chế độ 'rel' thay ràng buộc tuyệt đối bằng ràng buộc thứ hạng
        # giữa các lớp.
        if mode == 'rel':
            loss_cons_visual_cross = lambda_visual_cross * (
                relational_text_loss(sketch_feat, sketch_feat_fixed,
                                     text_distill_sketch, temperature)
                + relational_text_loss(photo_feat, photo_feat_fixed,
                                       text_distill_photo, temperature))
        else:
            loss_cons_visual_cross = lambda_visual_cross * (
                text_align_loss(sketch_feat, text_distill_sketch[label], mode)
                + text_align_loss(photo_feat, text_distill_photo[label], mode))
    else:
        loss_cons_text = 0.0
        loss_cons_visual_cross = 0.0

    # Total loss: L1 (cross-modal) + L2 (visual consistency) + L4 (CE) + L_cons_text + L_cons_visual_cross
    # Text distillation consistency remains disabled in this configuration.
    total_loss = loss_cross_modal + loss_consistency + loss_ce + loss_cons_text + loss_cons_visual_cross

    loss_dict = {
        'loss_cross_modal': loss_cross_modal,
        'loss_consistency': loss_consistency,
        'loss_ce': loss_ce,
        'loss_cons_text': loss_cons_text,
        'loss_cons_visual_cross': loss_cons_visual_cross
    }

    return total_loss, loss_dict
