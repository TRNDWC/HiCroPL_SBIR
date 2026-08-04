import copy
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision

from src.hicropl import (
    TextEncoder,
    VisualEncoder,
    VisualVisualPromptLearner,
    SimpleTextPromptLearner,
)
from src.losses_hicropl import loss_fn_hicropl, uses_triplet

def freeze_model(m):
    """Freeze all parameters of the given module."""
    for param in m.parameters():
        param.requires_grad_(False)
        

def freeze_all_but_bn(m):
    """
    DEPRECATED - KHÔNG DÙNG. Giữ lại chỉ để tương thích ngược.

    Hàm này KHÔNG đóng băng hết: `model.apply()` duyệt theo module và ở đây chỉ
    đụng tới `m.weight` / `m.bias`, nên mọi `nn.Parameter` khai báo trực tiếp vẫn
    còn requires_grad=True, cụ thể với CLIP ViT-B/32:
      - nn.MultiheadAttention.in_proj_weight / in_proj_bias (24 layer, ~30.7M params)
      - CLIP.text_projection, CLIP.positional_embedding, CLIP.logit_scale
      - visual.proj, visual.class_embedding, visual.positional_embedding
    Tổng cộng ~31.4M param leak mỗi bản CLIP, so với ~65K LayerNorm thực sự cần train.

    Dùng `freeze_model(m)` + `m.apply(unfreeze_ln)` thay thế.
    """
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad_(False)


def unfreeze_ln(m):
    """Mở lại weight/bias của mọi LayerNorm trong module.

    Dùng SAU `freeze_model(...)` để thực thi pattern "chỉ LN trainable":
        freeze_model(encoder)           # đông cứng tất cả
        encoder.apply(unfreeze_ln)      # chỉ mở LN
    """
    if isinstance(m, nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(True)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(True)

def _normalize_classname(name):
    return str(name).strip().lower().replace(" ", "_")


def _resolve_text_file(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def _load_gpt_distill_prompts(classnames, gpt_text_file):
    text_file = _resolve_text_file(gpt_text_file)
    if not text_file.exists():
        raise FileNotFoundError(f"GPT text file not found: {text_file}")

    with text_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    prompts_by_modality = {"photo": {}, "sketch": {}}
    for row in rows:
        cls = _normalize_classname(row.get("class", ""))
        input_text = str(row.get("input", "")).lower()
        output_text = str(row.get("output", "")).strip()
        if not cls or not output_text:
            continue
        if "sketch" in input_text:
            prompts_by_modality["sketch"][cls] = output_text
        elif "photo" in input_text:
            prompts_by_modality["photo"][cls] = output_text

    prompts = {"photo": [], "sketch": []}
    missing = {"photo": [], "sketch": []}
    for classname in classnames:
        key = _normalize_classname(classname)
        for modality in ("photo", "sketch"):
            prompt = prompts_by_modality[modality].get(key)
            if prompt is None:
                missing[modality].append(classname)
                prompt = f"a {modality} of a {str(classname).replace('_', ' ')}."
            prompts[modality].append(prompt)

    for modality, names in missing.items():
        if names:
            print(
                f"Warning: missing {len(names)} {modality} GPT prompts in {text_file}; "
                "falling back to template prompts."
            )

    return prompts


class CustomCLIP(nn.Module):
    """
    HiCroPL-SBIR Architecture Wrapper.
    Sử dụng HiCroPLFeatureExtractor làm nòng cốt.
    """

    def __init__(self, cfg, clip_model, clip_model_frozen, classnames=None):
        super().__init__()
        self.cfg = cfg
        
        if classnames is None:
            classnames = []
        if len(classnames) == 0:
            raise ValueError("CustomCLIP requires non-empty classnames during initialization.")

        original_device = next(clip_model.parameters()).device
        self.dtype = clip_model.dtype

        # 1. Branch-specific models (3 deep copies + 2 distill branches)
        self.clip_photo = copy.deepcopy(clip_model).to(original_device)
        self.clip_sketch = copy.deepcopy(clip_model).to(original_device)
        self.clip_distill_photo = copy.deepcopy(clip_model_frozen).to(original_device)
        self.clip_distill_sketch = copy.deepcopy(clip_model_frozen).to(original_device)

        # Backward-compatible alias for older code paths
        self.clip_distill = self.clip_distill_photo

        # -- Freeze policy --
        # Student: đóng băng toàn bộ rồi mở lại ĐÚNG LayerNorm.
        # `freeze_model` duyệt `.parameters()` (đệ quy, gồm cả bare nn.Parameter như
        # in_proj_weight / text_projection / proj / logit_scale) nên không sót gì,
        # khác với `freeze_all_but_bn` chỉ đụng `m.weight` / `m.bias`.
        for branch in (self.clip_photo, self.clip_sketch):
            freeze_model(branch)
            branch.apply(unfreeze_ln)

        # Teacher/distill: đóng băng HOÀN TOÀN, không chừa LayerNorm.
        # Nếu teacher còn trainable thì mục tiêu distill trôi theo student và các
        # loss dạng `1 - cos(feat + feat_distill, feat)` bị tối ưu bằng cách kéo
        # teacher về phía student -> collapse, loss giảm nhưng không học được gì.
        for teacher in (self.clip_distill_photo, self.clip_distill_sketch):
            freeze_model(teacher)
            teacher.eval()

        # Ảnh negative chỉ cần khi loss dùng triplet. Lấy vị từ từ losses_hicropl
        # để forward và loss không thể bất đồng về điều kiện này.
        self.needs_neg = uses_triplet(cfg)
        if not self.needs_neg:
            print("eval_mode=category: bỏ qua forward nhánh negative (không loss nào dùng)")

        # Nếu muốn học logit_scale, bật lại tường minh ở đây và nhớ clamp <= log(100)
        # trong forward. Mặc định giữ đóng băng cho đúng tinh thần prompt tuning.
        self.learn_logit_scale = bool(getattr(cfg, 'learn_logit_scale', False))
        if self.learn_logit_scale:
            self.clip_photo.logit_scale.requires_grad_(True)
            self.clip_sketch.logit_scale.requires_grad_(True)

        # Print trainable param counts per branch for verification
        def _count_trainable(m):
            total = 0
            trainable = 0
            for p in m.parameters():
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()
            return total, trainable

        for name, module in (
            ("clip_photo", self.clip_photo),
            ("clip_sketch", self.clip_sketch),
            ("clip_distill_photo", self.clip_distill_photo),
            ("clip_distill_sketch", self.clip_distill_sketch),
        ):
            tot, tr = _count_trainable(module)
            print(f"{name}: trainable {tr:,} / total {tot:,} params")

        # 3. Logit scales (unique to each prompted model)
        self.logit_scale_photo = self.clip_photo.logit_scale
        self.logit_scale_sketch = self.clip_sketch.logit_scale

        # -- Prompt Learners --
        # Initialize Visual-Visual learner + simple text learners + adapters
        print("Initializing Visual-Visual Prompt Learner (sketch <-> photo)...")
        self.visual_visual_learner = VisualVisualPromptLearner(cfg, self.clip_sketch, self.clip_photo)

        print("Initializing Photo Text Prompt Learner...")
        cfg_photo = copy.copy(cfg)
        cfg_photo.ctx_init = getattr(cfg, 'ctx_init', 'a photo of a')
        self.text_prompt_photo = SimpleTextPromptLearner(cfg_photo, classnames, self.clip_photo)

        print("Initializing Sketch Text Prompt Learner...")
        cfg_sketch = copy.copy(cfg)
        cfg_sketch.ctx_init = getattr(cfg, 'ctx_init_sketch', 'a sketch of a')
        self.text_prompt_sketch = SimpleTextPromptLearner(cfg_sketch, classnames, self.clip_sketch)

        # -- Encoders (Main Branches using their own models with ALL LNs open) --
        self.text_encoder_photo = TextEncoder(self.clip_photo)
        self.text_encoder_sketch = TextEncoder(self.clip_sketch)
        self.visual_encoder_photo = VisualEncoder(self.clip_photo)
        self.visual_encoder_sketch = VisualEncoder(self.clip_sketch)

        gpt_text_file = getattr(cfg, 'gpt_text_file', 'gpt_file/sketchy_ext.json')
        gpt_prompts = _load_gpt_distill_prompts(classnames, gpt_text_file)
        from src.clip import clip as _clip
        if classnames:
            self.register_buffer("tokenized_gpt_photo", _clip.tokenize(gpt_prompts["photo"], truncate=True))
            self.register_buffer("tokenized_gpt_sketch", _clip.tokenize(gpt_prompts["sketch"], truncate=True))
        else:
            self.register_buffer("tokenized_gpt_photo", torch.empty(0, 77, dtype=torch.long))
            self.register_buffer("tokenized_gpt_sketch", torch.empty(0, 77, dtype=torch.long))

        # -- Extractors removed: logic will be inlined in forward() --

    def train(self, mode=True):
        """Giữ 2 nhánh distill luôn ở eval mode.

        Lightning gọi `model.train()` trên toàn LightningModule mỗi epoch, ghi đè
        `clip_model_frozen.eval()` đã set ở script train. Override ở đây để teacher
        không bao giờ bị bật lại train mode.
        """
        super().train(mode)
        self.clip_distill_photo.eval()
        self.clip_distill_sketch.eval()
        return self

    def normalize_features(self, feat_prenorm):
        """L2-normalize feature tensors."""
        return feat_prenorm / feat_prenorm.norm(dim=-1, keepdim=True)

    def _exp_logit_scale(self, logit_scale_param):
        """exp(logit_scale) có clamp, theo đúng CLIP gốc (max = 100).

        Chỉ có tác dụng khi `learn_logit_scale=True`; mặc định logit_scale bị
        freeze nên clamp là no-op, nhưng giữ ở đây để nếu bật học thì scale
        không thể phân kỳ.
        """
        return logit_scale_param.clamp(max=float(np.log(100.0))).exp()

    def _encode_gpt_distill(self):
        """Encode GPT prompts của teacher, cache lại theo device.

        Teacher đã freeze hoàn toàn và tokenized_gpt_* là hằng số theo classname,
        nên feature không đổi giữa các step -> encode 1 lần thay vì 2 full
        text-encoder pass mỗi batch. Luôn chạy dưới no_grad.
        """
        cache = getattr(self, '_gpt_distill_cache', None)
        device = self.tokenized_gpt_photo.device
        if cache is not None and cache[0].device == device:
            return cache

        with torch.no_grad():
            text_distill_photo = self.clip_distill_photo.encode_text(self.tokenized_gpt_photo)
            text_distill_photo = text_distill_photo / text_distill_photo.norm(dim=-1, keepdim=True)

            text_distill_sketch = self.clip_distill_sketch.encode_text(self.tokenized_gpt_sketch)
            text_distill_sketch = text_distill_sketch / text_distill_sketch.norm(dim=-1, keepdim=True)

        self._gpt_distill_cache = (text_distill_photo, text_distill_sketch)
        return self._gpt_distill_cache

    def forward(self, x, classnames):
        """
        Forward pass for training with optimized redundancy.
        Calls visual learner ONCE and routes prompts by branch.
        """
        if len(x) == 5:
            sk_tensor, photo_tensor, neg_tensor, label, filename = x
            sk_aug_tensor = photo_aug_tensor = None
        elif len(x) == 7:
            sk_tensor, photo_tensor, neg_tensor, sk_aug_tensor, photo_aug_tensor, label, filename = x
        else:
            sk_tensor, photo_tensor, neg_tensor, sk_aug_tensor, photo_aug_tensor, label = x[:6]
        
        # 1. Call visual-visual learner ONCE (shared by both branches)
        vis1_shallow, vis2_shallow, vis1_deeper, vis2_deeper = self.visual_visual_learner()
        
        # 2. Photo branch: text learner + visual routing (vis2)
        # Compute text features for ALL classes (not just batch) - needed for loss computation
        text_input_photo_all, cross_prompts_text_deeper_photo = self.text_prompt_photo(label=None)  # All classes
        text_features_all_photo = self.text_encoder_photo(text_input_photo_all, self.text_prompt_photo.tokenized_prompts, cross_prompts_text_deeper_photo)
        image_features_photo = self.visual_encoder_photo(photo_tensor.type(self.dtype), vis2_shallow, vis2_deeper)
        out_p = {
            "image_features": image_features_photo,
            "text_features": text_features_all_photo,
            "text_features_all": text_features_all_photo,
            "logit_scale": self._exp_logit_scale(self.logit_scale_photo)
        }
        
        # 3. Sketch branch: text learner + visual routing (vis1)
        # Compute text features for ALL classes (not just batch) - needed for loss computation
        text_input_sketch_all, cross_prompts_text_deeper_sketch = self.text_prompt_sketch(label=None)  # All classes
        text_features_all_sketch = self.text_encoder_sketch(text_input_sketch_all, self.text_prompt_sketch.tokenized_prompts, cross_prompts_text_deeper_sketch)
        image_features_sketch = self.visual_encoder_sketch(sk_tensor.type(self.dtype), vis1_shallow, vis1_deeper)
        out_s = {
            "image_features": image_features_sketch,
            "text_features": text_features_all_sketch,
            "text_features_all": text_features_all_sketch,
            "logit_scale": self._exp_logit_scale(self.logit_scale_sketch)
        }
        
        # 4. Negative branch (uses photo encoder + photo visual prompts)
        # Chỉ chạy khi loss thật sự dùng tới: `neg_feat` chỉ vào L1 dạng triplet.
        # Ở eval_mode=category đây là một lượt forward+backward ViT đầy đủ hoàn
        # toàn bị vứt đi (~20-25% thời gian mỗi step).
        # KHÔNG bỏ neg_tensor khỏi dataset: dataloader tiêu RNG, bỏ đi sẽ đổi
        # chuỗi ngẫu nhiên và làm kết quả không so được với các run cũ. CLIP
        # không có dropout nên bỏ riêng forward pass cho kết quả giống hệt.
        if self.needs_neg:
            image_features_neg = self.visual_encoder_photo(neg_tensor.type(self.dtype), vis2_shallow, vis2_deeper)
            neg_feat = image_features_neg / image_features_neg.norm(dim=-1, keepdim=True)
        else:
            neg_feat = None

        # 2. Distill Visual Features (teacher đã freeze hoàn toàn) - RUN ONCE
        # Bọc no_grad: teacher chỉ là nguồn target, không được nhận gradient từ
        # residual mix (photo_feat_prenorm) hay từ L2 consistency loss.
        with torch.no_grad():
            if photo_aug_tensor is not None and sk_aug_tensor is not None:
                photo_aug_feat_fixed = self.clip_distill_photo.visual(photo_aug_tensor.type(self.dtype))
                photo_aug_feat_fixed = photo_aug_feat_fixed / photo_aug_feat_fixed.norm(dim=-1, keepdim=True)

                sketch_aug_feat_fixed = self.clip_distill_sketch.visual(sk_aug_tensor.type(self.dtype))
                sketch_aug_feat_fixed = sketch_aug_feat_fixed / sketch_aug_feat_fixed.norm(dim=-1, keepdim=True)
            else:
                photo_aug_feat_fixed = None
                sketch_aug_feat_fixed = None

            # Distill Visual Features for Original (for residual mix)
            photo_feat_fixed = self.clip_distill_photo.visual(photo_tensor.type(self.dtype))
            photo_feat_fixed = photo_feat_fixed / photo_feat_fixed.norm(dim=-1, keepdim=True)

            sketch_feat_fixed = self.clip_distill_sketch.visual(sk_tensor.type(self.dtype))
            sketch_feat_fixed = sketch_feat_fixed / sketch_feat_fixed.norm(dim=-1, keepdim=True)

        # 3. Residual Mix & Final Normalization
        # Image
        photo_feat_prompted = out_p["image_features"]
        photo_feat_prompted_norm = photo_feat_prompted / photo_feat_prompted.norm(dim=-1, keepdim=True)
        photo_feat_prenorm = photo_feat_prompted_norm + photo_feat_fixed
        photo_feat = photo_feat_prenorm / photo_feat_prenorm.norm(dim=-1, keepdim=True)

        sketch_feat_prompted = out_s["image_features"]
        sketch_feat_prompted_norm = sketch_feat_prompted / sketch_feat_prompted.norm(dim=-1, keepdim=True)
        sketch_feat_prenorm = sketch_feat_prompted_norm + sketch_feat_fixed
        sketch_feat = sketch_feat_prenorm / sketch_feat_prenorm.norm(dim=-1, keepdim=True)
        
        # neg_feat đã được chuẩn hoá ở bước 4 (hoặc là None nếu loss không dùng)

        text_feat_photo_prompted = out_p["text_features"]
        text_feat_photo = text_feat_photo_prompted / text_feat_photo_prompted.norm(dim=-1, keepdim=True)

        text_feat_sketch_prompted = out_s["text_features"]
        text_feat_sketch = text_feat_sketch_prompted / text_feat_sketch_prompted.norm(dim=-1, keepdim=True)

        # Encode GPT distill features for all classes (loss will select batch entries)
        text_distill_photo, text_distill_sketch = self._encode_gpt_distill()

        # 5. Compute Logits
        # Mỗi nhánh dùng logit_scale CỦA CHÍNH NÓ. Bản cũ lấy `out_p["logit_scale"]`
        # cho cả hai, nên `logit_scale_sketch` được tính rồi vứt: nó không nằm
        # trong đồ thị, grad=None, và khi bật --learn_logit_scale thì optimizer
        # nhận một param không bao giờ được cập nhật.
        scale_photo = out_p["logit_scale"]
        scale_sketch = out_s["logit_scale"]
        logits_photo = scale_photo * photo_feat @ text_feat_photo.t()
        logits_sketch = scale_sketch * sketch_feat @ text_feat_sketch.t()

        # Logits for Augmented Images
        if photo_aug_feat_fixed is not None and sketch_aug_feat_fixed is not None:
            logits_photo_aug = scale_photo * photo_aug_feat_fixed @ text_feat_photo.t()
            logits_sketch_aug = scale_sketch * sketch_aug_feat_fixed @ text_feat_sketch.t()
        else:
            logits_photo_aug = None
            logits_sketch_aug = None
        
        return (
            photo_feat, logits_photo,
            sketch_feat, logits_sketch,
            neg_feat, label,
            photo_aug_feat_fixed, sketch_aug_feat_fixed,
            logits_photo_aug, logits_sketch_aug,
            text_feat_photo, text_feat_sketch,
            text_distill_photo, text_distill_sketch,
            photo_feat_fixed, sketch_feat_fixed,
        )


class HiCroPL_SBIR(pl.LightningModule):
    def __init__(self, cfg, args, classnames, model):
        super().__init__()
        self.cfg = cfg
        self.args = args
        self.classnames = classnames
        self.model = model
        
        self.best_metric = 1e-3
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)

        self.test_photo_features = []
        self.test_sketch_features = []
        self.test_photo_labels = []
        self.test_sketch_labels = []

    def on_train_epoch_start(self):
        # NOTE: Encoders stay in training mode (required for LayerNorm to use batch statistics)
        # Setting eval() here would conflict with forward() expectation and break BN/LN behavior
        pass

    def on_fit_start(self):
        """Log the number of learnable prompt tokens per branch once at fit start.

        Logs four scalars (tokens count):
        - `tokens_visual_photo`
        - `tokens_visual_sketch`
        - `tokens_text_photo`
        - `tokens_text_sketch`
        """
        try:
            vv = self.model.visual_visual_learner
            # visual tokens: number of prompt vectors (prompt_depth * n_ctx)
            tokens_visual_photo = len(vv.cross_prompts_photo) * vv.n_ctx
            tokens_visual_sketch = len(vv.cross_prompts_sketch) * vv.n_ctx
        except Exception:
            tokens_visual_photo = 0
            tokens_visual_sketch = 0

        try:
            tp = self.model.text_prompt_photo
            ts = self.model.text_prompt_sketch
            tokens_text_photo = len(tp.cross_prompts_text) * tp.cross_prompts_text[0].shape[0]
            tokens_text_sketch = len(ts.cross_prompts_text) * ts.cross_prompts_text[0].shape[0]
        except Exception:
            tokens_text_photo = 0
            tokens_text_sketch = 0

        # Log to Lightning logger and print for immediate visibility
        self.print(f"Learnable tokens - visual/photo: {tokens_visual_photo}, visual/sketch: {tokens_visual_sketch}, text/photo: {tokens_text_photo}, text/sketch: {tokens_text_sketch}")

        self._assert_no_param_leak()
        # Use self.log so TensorBoard/other loggers capture these scalars
        # Use rank_zero_only to avoid duplicate logs in distributed runs
        try:
            self.log('tokens_visual_photo', tokens_visual_photo, prog_bar=True, logger=True)
            self.log('tokens_visual_sketch', tokens_visual_sketch, prog_bar=True, logger=True)
            self.log('tokens_text_photo', tokens_text_photo, prog_bar=True, logger=True)
            self.log('tokens_text_sketch', tokens_text_sketch, prog_bar=True, logger=True)
        except Exception:
            # Fallback to print-only if logger not ready
            pass

    def _assert_no_param_leak(self):
        """Fail sớm nếu freeze policy bị rò rỉ, chạy ở `on_fit_start`.

        Hai bất biến:
          1. Không param nào của teacher (clip_distill_*) được trainable.
          2. Trong backbone student, nguồn trainable hợp lệ DUY NHẤT là LayerNorm
             (cộng logit_scale nếu bật `learn_logit_scale`). Mọi thứ khác -
             in_proj_weight, text_projection, visual.proj, positional_embedding,
             class_embedding - đều là dấu hiệu freeze policy bị thủng.

        Param của prompt learner được loại trừ theo tên: chúng có
        nn.MultiheadAttention riêng và đúng là phải trainable.
        """
        model = self.model
        learner_prefixes = ('visual_visual_learner', 'text_prompt_photo', 'text_prompt_sketch')
        distill_prefixes = ('clip_distill_photo', 'clip_distill_sketch', 'clip_distill')

        ln_ids = {
            id(p)
            for _, m in model.named_modules() if isinstance(m, torch.nn.LayerNorm)
            for p in m.parameters(recurse=False)
        }
        allowed_ids = set()
        if getattr(model, 'learn_logit_scale', False):
            allowed_ids = {id(model.clip_photo.logit_scale), id(model.clip_sketch.logit_scale)}

        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.print(
            f"TRAINABLE: {len(trainable)} tensors, "
            f"{sum(p.numel() for _, p in trainable):,} params"
        )

        problems = []

        distill = [n for n, _ in trainable if n.startswith(distill_prefixes)]
        if distill:
            problems.append(
                f"teacher chưa freeze ({len(distill)} tensors): {distill[:5]}"
            )

        backbone = [
            n for n, p in trainable
            if not n.startswith(learner_prefixes)
            and not n.startswith(distill_prefixes)
            and id(p) not in ln_ids
            and id(p) not in allowed_ids
        ]
        if backbone:
            problems.append(
                f"param backbone ngoài LayerNorm vẫn trainable ({len(backbone)} tensors): {backbone[:10]}"
            )

        if problems:
            raise RuntimeError("Phát hiện leak parameters:\n  - " + "\n  - ".join(problems))

    def configure_optimizers(self):
        def add_unique_params(candidates, out_list, seen_ids):
            for p in candidates:
                if p.requires_grad and id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    out_list.append(p)

        seen_ids = set()

        prompt_params = []
        # Collect from shared visual learner + per-branch text learners
        # These include all their internal params (CrossPromptAttention, AttentionPooling, etc.)
        add_unique_params(self.model.visual_visual_learner.parameters(), prompt_params, seen_ids)
        add_unique_params(self.model.text_prompt_photo.parameters(), prompt_params, seen_ids)
        add_unique_params(self.model.text_prompt_sketch.parameters(), prompt_params, seen_ids)

        ln_params = []
        # Only collect LayerNorms from clip encoders (NOT from learners, already included above)
        learner_modules = {
            'visual_visual_learner', 'text_prompt_photo', 'text_prompt_sketch'
        }
        # Teacher/distill phải nằm ngoài optimizer hoàn toàn. Filter cũ chỉ loại
        # learner module nên LayerNorm của clip_distill_* vẫn lọt vào param group.
        distill_prefixes = ('clip_distill_photo', 'clip_distill_sketch', 'clip_distill')
        for name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.LayerNorm):
                continue
            if name.startswith(distill_prefixes):
                continue
            # Skip if inside a learner module (already included with learner params)
            if any(learner_name in name for learner_name in learner_modules):
                continue
            add_unique_params(module.parameters(recurse=False), ln_params, seen_ids)

        # logit_scale chỉ vào optimizer khi được bật tường minh (cfg.learn_logit_scale)
        if getattr(self.model, 'learn_logit_scale', False):
            add_unique_params(
                [self.model.clip_photo.logit_scale, self.model.clip_sketch.logit_scale],
                ln_params, seen_ids,
            )

        # KHÔNG dùng catch-all "gom mọi param requires_grad còn sót" nữa: đó chính là
        # chỗ biến leak của freeze policy thành leak thật trong optimizer. Thay bằng
        # kiểm tra - nếu còn param trainable nào không thuộc whitelist thì fail sớm.
        leaked = [
            n for n, p in self.model.named_parameters()
            if p.requires_grad and id(p) not in seen_ids
        ]
        if leaked:
            raise RuntimeError(
                f"{len(leaked)} param trainable nằm ngoài whitelist (prompt learners + LayerNorm "
                f"của student). Kiểm tra lại freeze policy trong CustomCLIP.__init__. "
                f"Ví dụ: {leaked[:10]}"
            )

        non_prompt_params = ln_params

        self.print(f"Number of trainable prompt params: {sum(p.numel() for p in prompt_params):,}")
        self.print(f"Number of trainable non-prompt params: {sum(p.numel() for p in non_prompt_params):,}")

        prompt_lr = getattr(self.cfg, 'prompt_lr', 1e-5)
        clip_ln_lr = getattr(self.cfg, 'clip_LN_lr', 1e-5)
        weight_decay = getattr(self.cfg, 'weight_decay', 1e-4)

        param_groups = [{'params': prompt_params, 'lr': prompt_lr}]
        if non_prompt_params:
            param_groups.append({'params': non_prompt_params, 'lr': clip_ln_lr})

        return torch.optim.Adam(param_groups, weight_decay=weight_decay)

    def training_step(self, batch, batch_idx):
        features = self.model(batch, self.classnames)
        loss, loss_dict = loss_fn_hicropl(self.args, features)
        
        # Log total loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        
        # Log individual loss components to TensorBoard and Terminal (prog_bar)
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor) or v > 0:
                self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                
        return loss

    def extract_eval_features(self, tensor, modality):
        """Extract visual features: Prompted + Distill Fixed (Residual Mix)"""
        # Chẩn đoán: bỏ hẳn nhánh prompted, chỉ dùng CLIP đóng băng. Cho biết
        # prompt thực sự đóng góp bao nhiêu điểm so với zero-shot thuần.
        if getattr(self.cfg, 'eval_frozen_only', False):
            distill = (self.model.clip_distill_photo if modality == 'photo'
                       else self.model.clip_distill_sketch).visual
            fixed = distill(tensor.type(self.model.dtype))
            return fixed / fixed.norm(dim=-1, keepdim=True)

        # Call visual learner once, cache outputs
        vis1_shallow, vis2_shallow, vis1_deeper, vis2_deeper = self.model.visual_visual_learner()

        if modality == 'photo':
            visual_encoder = self.model.visual_encoder_photo
            distill_encoder = self.model.clip_distill_photo.visual
            vis_shallow, vis_deeper = vis2_shallow, vis2_deeper
        else:
            visual_encoder = self.model.visual_encoder_sketch
            distill_encoder = self.model.clip_distill_sketch.visual
            vis_shallow, vis_deeper = vis1_shallow, vis1_deeper

        # Retrieval theo ảnh nên không cần text feature ở đây. Lời gọi
        # `text_learner(label=None)` cũ bị vứt kết quả nhưng vẫn cấp phát
        # [n_cls, 77, 512] mỗi val batch. Prompt learner đã thuần hoá (không còn
        # `.data.copy_()`) nên bỏ nó đi không đổi trạng thái model.
        prompted_feat = visual_encoder(tensor.type(self.model.dtype), vis_shallow, vis_deeper)
        prompted_feat_norm = prompted_feat / prompted_feat.norm(dim=-1, keepdim=True)
        
        fixed_feat = distill_encoder(tensor.type(self.model.dtype))
        fixed_feat_norm = fixed_feat / fixed_feat.norm(dim=-1, keepdim=True)
        
        combined_prenorm = prompted_feat_norm + fixed_feat_norm
        return combined_prenorm / combined_prenorm.norm(dim=-1, keepdim=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._validation_step_category(batch, batch_idx, dataloader_idx)

    def _validation_step_category(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 3:
            tensor, label, type_data = batch
        else:
            tensor, label = batch
            type_data = None
            
        if dataloader_idx == 0:
            sketch_feat = self.extract_eval_features(tensor, modality='sketch')
            self.test_sketch_features.append(sketch_feat.cpu().detach()) 
            self.test_sketch_labels.append(label.cpu().detach())
        elif dataloader_idx == 1:
            photo_feat = self.extract_eval_features(tensor, modality='photo')
            self.test_photo_features.append(photo_feat.cpu().detach())   
            self.test_photo_labels.append(label.cpu().detach())

    def on_validation_epoch_end(self):
        return self._on_validation_epoch_end_category()

    def _on_validation_epoch_end_category(self):
        if not self.test_photo_features or not self.test_sketch_features:
            self.print("Warning: Missing features for validation. Skipping metrics.")
            return

        gallery_features = torch.cat(self.test_photo_features, dim=0).to(self.device)
        query_features   = torch.cat(self.test_sketch_features, dim=0).to(self.device)
        
        all_photo_category  = torch.cat(self.test_photo_labels, dim=0).to(self.device)
        all_sketch_category = torch.cat(self.test_sketch_labels, dim=0).to(self.device)

        similarity_matrix = query_features @ gallery_features.t()

        dataset = getattr(self.args, 'dataset', 'sketchy')
        if dataset == "sketchy_2" or dataset == "sketchy_ext":
            map_k = 200
            p_k = 200
        elif dataset == "quickdraw":
            map_k = 0
            p_k = 200
        else:
            map_k = 0
            p_k = 100

        ap = torch.zeros(len(query_features), device=self.device)
        precision = torch.zeros(len(query_features), device=self.device)

        for idx in range(len(query_features)):
            category = all_sketch_category[idx]
            distance = similarity_matrix[idx]
            target = (all_photo_category == category)

            if map_k != 0:
                top_k_actual = min(map_k, len(gallery_features))
                ap[idx] = retrieval_average_precision(distance, target, top_k=top_k_actual)
            else:
                ap[idx] = retrieval_average_precision(distance, target)

            precision[idx] = retrieval_precision(distance, target, top_k=p_k)

        mAP = torch.mean(ap)
        mean_precision = torch.mean(precision)

        self.log("mAP", mAP, on_step=False, on_epoch=True)
        self.log(f"P@{p_k}", mean_precision, on_step=False, on_epoch=True)
        self.log("val_mAP", mAP, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"val_P@{p_k}", mean_precision, on_step=False, on_epoch=True)

        if map_k != 0:
            self.log(f"val_map_{map_k}", mAP, on_step=False, on_epoch=True)
        else:
            self.log("val_map_all", mAP, on_step=False, on_epoch=True)
        self.log(f"val_p_{p_k}", mean_precision, on_step=False, on_epoch=True)

        is_new_best = False
        if self.global_step > 0:
            if mAP.item() > self.best_metric:
                self.best_metric = mAP.item()
                is_new_best = True

        # Log SAU khi cập nhật: thứ tự cũ ghi giá trị của epoch trước nên biểu đồ
        # TensorBoard luôn trễ một epoch.
        self.log("best_mAP", self.best_metric, on_step=False, on_epoch=True, prog_bar=False)

        if is_new_best:
            self._save_ap_vector(ap, precision, all_sketch_category, map_k, p_k)

        if map_k != 0:
            self.print('mAP@{}: {:.4f}, P@{}: {:.4f}, Best mAP: {:.4f}'.format(
                map_k, mAP.item(), p_k, mean_precision.item(), self.best_metric))
        else:
            self.print('mAP@all: {:.4f}, P@{}: {:.4f}, Best mAP: {:.4f}'.format(
                mAP.item(), p_k, mean_precision.item(), self.best_metric))

        train_loss = self.trainer.callback_metrics.get("train_loss", None)
        if train_loss is not None:
            self.print(f"Train loss (epoch avg): {train_loss.item():.6f}")

        self.test_photo_features.clear()
        self.test_sketch_features.clear()
        self.test_photo_labels.clear()
        self.test_sketch_labels.clear()

    def _save_ap_vector(self, ap, precision, sketch_labels, map_k, p_k):
        """Lưu AP từng query ở epoch tốt nhất, để kiểm cặp giữa hai run.

        So hai giá trị mAP trung bình rất kém nhạy: sai số chuẩn trên 12,694
        query cỡ 0.2 pp. Nhưng hai model được đánh giá trên CÙNG tập query theo
        cùng thứ tự (val loader shuffle=False), nên so theo cặp từng query khử
        được phần lớn phương sai và phát hiện được chênh lệch nhỏ hơn nhiều.

        `sketch_labels` được lưu kèm để scripts/paired_test.py xác nhận hai run
        thực sự cùng thứ tự query trước khi so.
        """
        run_dir = getattr(self.cfg, 'run_dir', None)
        if not run_dir:
            return
        try:
            out = Path(run_dir) / 'ap_best.npz'
            np.savez(
                out,
                ap=ap.detach().cpu().numpy(),
                precision=precision.detach().cpu().numpy(),
                sketch_labels=sketch_labels.detach().cpu().numpy(),
                epoch=np.array(self.current_epoch),
                best_map=np.array(self.best_metric),
                map_k=np.array(map_k),
                p_k=np.array(p_k),
            )
        except Exception as e:
            self.print(f'Không lưu được ap_best.npz: {type(e).__name__}: {e}')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
