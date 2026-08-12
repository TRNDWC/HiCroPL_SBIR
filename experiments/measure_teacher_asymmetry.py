"""
Đo trực tiếp năng lực "teacher" của CLIP ViT-B/32 hoàn toàn đóng băng (KHÔNG
prompt injection nào -- vanilla CLIP, giống hệt kỹ thuật clip_trainer='CoOp'
đã dùng cho --no_prompt_learning trong src/model_hicropl.py) trên 21 unseen
categories (UNSEEN_CLASSES['sketchy'], src/dataset_retrieval.py), tách riêng
photo và sketch -- phục vụ kiểm định H1 (Asymmetric Teacher Hypothesis).

Zero-shot classification chuẩn CLIP: với mỗi ảnh, tính cosine similarity giữa
image embedding và text embedding của N template ("a photo of a {c}." cho
ảnh photo, "a sketch of a {c}." cho ảnh sketch), argmax -> predicted class,
so với ground-truth category -> accuracy.

KHÔNG train, KHÔNG dùng bất kỳ prompt learner nào trong src/hicropl.py --
đây là con số zero-shot THUẦN của CLIP gốc, đóng vai trò "năng lực teacher"
mà HiCroPL/VisualVisualPromptLearner giả định ngầm là ngang nhau giữa 2 domain.

Chạy:
    python -m experiments.measure_teacher_asymmetry --data_dir /path/to/Sketchy
"""
import argparse
import torch
from torch.utils.data import DataLoader

from src.clip import clip as _clip
from src.dataset_retrieval import ValidDataset, UNSEEN_CLASSES

parser = argparse.ArgumentParser(description='H1 teacher-asymmetry probe')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--dataset', type=str, default='sketchy')
parser.add_argument('--backbone', type=str, default='ViT-B/32')
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--workers', type=int, default=4)
opts = parser.parse_args()


@torch.no_grad()
def zero_shot_accuracy(clip_model, loader, classnames, template, device):
    """template: e.g. "a photo of a {}." """
    prompts = [template.format(c.replace('_', ' ')) for c in classnames]
    tokenized = _clip.tokenize(prompts).to(device)
    text_feat = clip_model.encode_text(tokenized)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        img_feat = clip_model.encode_image(images.type(clip_model.dtype))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        pred = (img_feat.float() @ text_feat.float().t()).argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / total, total


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Vanilla CLIP (design_details['trainer']='CoOp' -> plain VisionTransformer/
    # ResidualAttentionBlock trong src/clip/model.py, KHÔNG injection nào) --
    # đúng kỹ thuật đã validate an toàn cho --no_prompt_learning.
    clip_model, _ = _clip.load(
        opts.backbone, device=device,
        design_details={'trainer': 'CoOp', 'vision_depth': 0, 'language_depth': 0,
                         'vision_ctx': 0, 'language_ctx': 0},
    )
    clip_model = clip_model.float().eval()

    unseen_classes = UNSEEN_CLASSES.get(opts.dataset, UNSEEN_CLASSES['sketchy'])
    print(f"[CONFIG] {len(unseen_classes)} unseen categories: {unseen_classes}")

    val_photo = ValidDataset(opts, mode='photo')
    val_sketch = ValidDataset(opts, mode='sketch')
    assert val_photo.all_categories == val_sketch.all_categories, \
        "CRITICAL: photo/sketch category order mismatch"
    classnames = val_photo.all_categories
    print(f"[CONFIG] photo images: {len(val_photo)}, sketch images: {len(val_sketch)}")

    photo_loader = DataLoader(val_photo, batch_size=opts.test_batch_size,
                               num_workers=opts.workers, shuffle=False)
    sketch_loader = DataLoader(val_sketch, batch_size=opts.test_batch_size,
                                num_workers=opts.workers, shuffle=False)

    acc_photo, n_photo = zero_shot_accuracy(
        clip_model, photo_loader, classnames, "a photo of a {}.", device
    )
    acc_sketch, n_sketch = zero_shot_accuracy(
        clip_model, sketch_loader, classnames, "a sketch of a {}.", device
    )

    print()
    print(f"Zero-shot accuracy (frozen CLIP {opts.backbone}, {len(classnames)} unseen categories):")
    print(f"  Photo  : {acc_photo:.4f}  (n={n_photo})")
    print(f"  Sketch : {acc_sketch:.4f}  (n={n_sketch})")
    print(f"  Gap (photo - sketch): {acc_photo - acc_sketch:+.4f}")
