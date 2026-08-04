# Sửa lỗi rò rỉ tham số (parameter leak) trong HiCroPL-SBIR

Tài liệu này mô tả năm chỗ rò rỉ tham số trong vòng huấn luyện, vì sao chúng
xảy ra, đã sửa như thế nào, và cách tự kiểm chứng.

Mô hình dùng làm ví dụ xuyên suốt: **CLIP ViT-B/32**, cấu hình
`n_ctx=2, prompt_depth=6, language_depth=1, cross_layer=3, eval_mode=category`.

---

## 1. Tóm tắt

Ý đồ thiết kế là **prompt tuning**: đóng băng CLIP, chỉ học prompt token, vài
mạng ánh xạ nhỏ, và LayerNorm của hai nhánh student. Nhưng thực tế:

| | Trước | Sau |
|---|---:|---:|
| Trainable params | **158,002,692** | **32,068,096** |
| Backbone bị rò rỉ vào optimizer | 125,803,524 | 0 |
| Teacher (distill) trainable | 63,032,834 | **0** |
| Gradient tới knowledge mapper | **không có** | có |
| Param bị mutate ngoài `optimizer.step()` | có | không |

Hai con số quan trọng nhất:

- **125,803,524 tham số backbone** đang được fine-tune trong khi tưởng là đóng băng.
- **31.9M tham số của knowledge mapper** nằm trong optimizer nhưng **chưa từng nhận
  một gradient nào** — cơ chế đóng góp chính của HiCroPL đứng yên ở giá trị khởi tạo
  suốt quá trình huấn luyện.

| # | Rò rỉ | File |
|---|---|---|
| A | `freeze_all_but_bn` bỏ sót mọi `nn.Parameter` khai báo trực tiếp | `src/model_hicropl.py` |
| B | Teacher/distill không được đóng băng, không ở eval mode, không có `no_grad` | `src/model_hicropl.py` |
| C | `configure_optimizers` có catch-all gom mọi param còn `requires_grad` | `src/model_hicropl.py` |
| D | `.data.copy_()` cắt đồ thị tính toán và mutate param ngoài optimizer | `src/hicropl.py` |
| E | Prompt learner giữ tham chiếu tới backbone dưới dạng submodule | `src/hicropl.py` |

---

## 2. Bối cảnh: cái gì lẽ ra phải học?

`CustomCLIP` giữ **bốn bản CLIP ViT-B/32 độc lập**:

| Module | Vai trò | Ý đồ |
|---|---|---|
| `clip_photo` | student nhánh ảnh | chỉ LayerNorm trainable |
| `clip_sketch` | student nhánh sketch | chỉ LayerNorm trainable |
| `clip_distill_photo` | teacher ảnh | **đóng băng tuyệt đối** |
| `clip_distill_sketch` | teacher sketch | **đóng băng tuyệt đối** |

Cộng ba prompt learner: `visual_visual_learner`, `text_prompt_photo`,
`text_prompt_sketch`.

Tập trainable **đúng** phải gồm:

```
LayerNorm của 2 student     131,072
prompt token (visual+text)   20,480
knowledge mapper + LKP   31,916,544
─────────────────────────────────────
                         32,068,096
```

Con số LayerNorm kiểm chứng được: ViT-B/32 có text tower 12 layer rộng 512
(`ln_1`, `ln_2` mỗi layer, cộng `ln_final`) và visual tower 12 layer rộng 768
(`ln_1`, `ln_2` mỗi layer, cộng `ln_pre`, `ln_post`):

```
text   : 12 × 2 × (2×512) + 2×512           = 25,600
visual : 12 × 2 × (2×768) + 2 × (2×768)     = 39,936
                                     tổng   = 65,536  (mỗi bản CLIP)
```

Log thực tế sau khi sửa in ra đúng `clip_photo: trainable 65,536` — khớp tuyệt đối.

---

## 3. Rò rỉ A — `freeze_all_but_bn` bỏ sót bare `nn.Parameter`

### 3.1 Code có vấn đề

```python
def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad_(False)

self.clip_photo.apply(freeze_all_but_bn)     # tưởng là đóng băng hết trừ LN
```

### 3.2 Vì sao sai

`nn.Module.apply(fn)` duyệt theo **module**, và `fn` ở đây chỉ đụng đúng hai
attribute có tên `weight` và `bias`. Mọi tham số khai báo trực tiếp dưới dạng
`nn.Parameter` với tên khác đều **không bao giờ được chạm tới**, nên giữ nguyên
`requires_grad=True` mặc định.

Ví dụ tối thiểu — chạy đoạn này để thấy vấn đề:

```python
import torch.nn as nn

def freeze_all_but_bn(m):
    if not isinstance(m, nn.LayerNorm):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad_(False)

block = nn.MultiheadAttention(768, 8)
block.apply(freeze_all_but_bn)
for n, p in block.named_parameters():
    print(f"{n:16} requires_grad={p.requires_grad}  ({p.numel():,} params)")
```

`nn.MultiheadAttention` **không có** attribute `weight` hay `bias` — nó dùng
`in_proj_weight` / `in_proj_bias`, còn `out_proj` là một submodule `Linear`
riêng. Nên `hasattr(m, "weight")` trả `False`, hàm không làm gì cả, và
`in_proj_weight` (1,769,472 params ở 768 chiều) vẫn trainable. Chỉ `out_proj`
bị đóng băng vì `apply` ghé thăm nó như một module riêng — tức là **đúng một nửa
khối attention**.

### 3.3 Toàn bộ danh sách bị bỏ sót

Mỗi bản CLIP ViT-B/32:

| Tham số | Khai báo tại | Params |
|---|---|---:|
| `in_proj_weight` + `in_proj_bias`, text tower (12 layer) | `nn.MultiheadAttention` | 9,455,616 |
| `in_proj_weight` + `in_proj_bias`, visual tower (12 layer) | `nn.MultiheadAttention` | 21,261,312 |
| `visual.proj` | `model.py:520` | 393,216 |
| `text_projection` | `model.py:687` | 262,144 |
| `positional_embedding` (text) | `model.py:684` | 39,424 |
| `visual.positional_embedding` | `model.py:511` | 38,400 |
| `visual.class_embedding` | `model.py:510` | 768 |
| `logit_scale` | `model.py:688` | 1 |
| | **mỗi bản CLIP** | **31,450,881** |
| | **× 4 bản** | **125,803,524** |

**480 lần** nhiều hơn 65,536 LayerNorm thực sự cần train.

### 3.4 Cách sửa

Trong file đã sẵn có hai hàm đúng nhưng chưa từng được gọi. Chỉ cần dùng chúng:

```python
def freeze_model(m):
    for param in m.parameters():        # đệ quy, bắt cả bare nn.Parameter
        param.requires_grad_(False)

def unfreeze_ln(m):
    if isinstance(m, nn.LayerNorm):
        m.weight.requires_grad_(True)
        m.bias.requires_grad_(True)
```

Thay thế:

```python
# TRƯỚC
self.clip_sketch.apply(freeze_all_but_bn)
self.clip_photo.apply(freeze_all_but_bn)
self.clip_distill_photo.apply(freeze_all_but_bn)
self.clip_distill_sketch.apply(freeze_all_but_bn)

# SAU
for branch in (self.clip_photo, self.clip_sketch):
    freeze_model(branch)          # đóng băng tất cả, không sót
    branch.apply(unfreeze_ln)     # rồi mở lại đúng LayerNorm

for teacher in (self.clip_distill_photo, self.clip_distill_sketch):
    freeze_model(teacher)         # KHÔNG chừa LayerNorm
    teacher.eval()
```

Điểm mấu chốt: `freeze_model` duyệt `.parameters()` — vốn đã đệ quy qua toàn bộ
cây module **và** trả về mọi `nn.Parameter` bất kể tên. Không thể sót.

`freeze_all_but_bn` được giữ lại để tương thích ngược nhưng đánh dấu
`DEPRECATED` kèm docstring liệt kê những gì nó bỏ sót.

Nếu muốn học `logit_scale` thì phải bật tường minh qua `--learn_logit_scale`,
và nó được clamp `≤ log(100)` như CLIP gốc để scale không phân kỳ.

---

## 4. Rò rỉ B — Teacher không được cô lập

### 4.1 Vấn đề

Teacher (`clip_distill_*`) là **nguồn mục tiêu** cho distillation và cho residual
mix. Nó phải là hằng số. Nhưng có ba đường khiến nó thay đổi, cần chặn cả ba:

| Đường | Triệu chứng | Cách chặn |
|---|---|---|
| `requires_grad=True` | LayerNorm teacher được optimizer cập nhật | `freeze_model` (mục 3.4) |
| train mode | Lightning gọi `model.train()` mỗi epoch, ghi đè `clip_model_frozen.eval()` đặt trong script | override `CustomCLIP.train()` |
| gradient chảy ngược | teacher nằm trong đồ thị của residual mix và L2 | `torch.no_grad()` |

### 4.2 Hậu quả nếu không sửa

Loss text consistency có dạng:

```python
loss = 1.0 - F.cosine_similarity(text_feat + text_distill, text_feat, dim=-1)
```

Nếu `text_distill` (teacher) học được, cách rẻ nhất để giảm loss **không phải**
là kéo student về phía teacher, mà là kéo **teacher** về cùng hướng student. Mục
tiêu distill tự trôi theo — loss giảm đẹp trên biểu đồ mà mô hình không học được
gì. Đây là dạng lỗi nguy hiểm vì nó không gây crash và không có cảnh báo.

Tương tự, `photo_feat_fixed` được cộng vào residual mix nên gradient chảy thẳng
vào visual tower của teacher.

### 4.3 Cách sửa

```python
class CustomCLIP(nn.Module):
    def train(self, mode=True):
        """Giữ 2 nhánh distill luôn ở eval mode.

        Lightning gọi model.train() trên toàn LightningModule mỗi epoch, ghi đè
        clip_model_frozen.eval() đã set ở script train.
        """
        super().train(mode)
        self.clip_distill_photo.eval()
        self.clip_distill_sketch.eval()
        return self
```

và bọc mọi lời gọi teacher:

```python
# TRƯỚC
photo_feat_fixed = self.clip_distill_photo.visual(photo_tensor.type(self.dtype))
photo_feat_fixed = photo_feat_fixed / photo_feat_fixed.norm(dim=-1, keepdim=True)

# SAU
with torch.no_grad():
    photo_feat_fixed = self.clip_distill_photo.visual(photo_tensor.type(self.dtype))
    photo_feat_fixed = photo_feat_fixed / photo_feat_fixed.norm(dim=-1, keepdim=True)
```

Nhân tiện: đặc trưng text GPT của teacher là **hằng số theo classname** (teacher
đã đóng băng, `tokenized_gpt_*` là buffer). Trước đây nó được encode lại mỗi
batch — hai lượt text encoder đầy đủ bị lãng phí. Nay cache theo device qua
`_encode_gpt_distill()`.

---

## 5. Rò rỉ C — Catch-all trong `configure_optimizers`

### 5.1 Code có vấn đề

```python
ln_params = []
learner_modules = {'visual_visual_learner', 'text_prompt_photo', 'text_prompt_sketch'}
for name, module in self.model.named_modules():
    if isinstance(module, torch.nn.LayerNorm):
        if not any(l in name for l in learner_modules):
            add_unique_params(module.parameters(recurse=False), ln_params, seen_ids)
            # ^ KHÔNG loại clip_distill_* -> LayerNorm của teacher lọt vào

extra_trainable_params = []
for _, p in self.model.named_parameters():
    if p.requires_grad and id(p) not in seen_ids:
        extra_trainable_params.append(p)
        # ^ hút TOÀN BỘ 125.8M rò rỉ ở mục A vào Adam
```

### 5.2 Vì sao đây là chỗ nguy hiểm nhất

Rò rỉ A chỉ tạo ra các tensor có `requires_grad=True`. Bản thân điều đó chưa cập
nhật gì. Chính vòng catch-all này mới **biến nó thành fine-tuning thật sự** bằng
cách đưa hết vào Adam với `clip_LN_lr`.

Vòng lặp được viết với ý tốt — "gom nốt những gì còn sót" — nhưng nó biến mọi
sai sót của freeze policy thành hành vi âm thầm thay vì lỗi.

### 5.3 Cách sửa: whitelist + fail-fast

```python
distill_prefixes = ('clip_distill_photo', 'clip_distill_sketch', 'clip_distill')
for name, module in self.model.named_modules():
    if not isinstance(module, torch.nn.LayerNorm):
        continue
    if name.startswith(distill_prefixes):        # MỚI: loại teacher
        continue
    if any(l in name for l in learner_modules):
        continue
    add_unique_params(module.parameters(recurse=False), ln_params, seen_ids)

# Bỏ hẳn catch-all, thay bằng kiểm tra
leaked = [n for n, p in self.model.named_parameters()
          if p.requires_grad and id(p) not in seen_ids]
if leaked:
    raise RuntimeError(
        f"{len(leaked)} param trainable nằm ngoài whitelist. "
        f"Kiểm tra lại freeze policy. Ví dụ: {leaked[:10]}")
```

Nguyên tắc: **optimizer không tự sửa lỗi cho freeze policy**. Nếu có gì đó không
khớp, hỏng ngay từ dòng đầu với tên param cụ thể, thay vì âm thầm train 126M
tham số suốt 60 epoch.

---

## 6. Rò rỉ D — `.data.copy_()` cắt gradient và mutate param

### 6.1 Code có vấn đề

Trong `VisualVisualPromptLearner.forward()` (và `CrossModalPromptLearner.forward()`):

```python
updated_sketch_prompts = self.photo2sketch_net(sketch_prompts_flat,
                                               proxy_photo_flat, proxy_photo_flat)
for i in range(self.cross_layer):
    self.cross_prompts_sketch[i].data.copy_(updated_sketch_prompts[i])
...
cross_prompts_sketch_deeper = [self.cross_prompts_sketch[i]
                               for i in range(1, len(self.cross_prompts_sketch))]
return ..., cross_prompts_sketch_deeper
```

Ý đồ: "prompt của sketch = kết quả ánh xạ từ prompt photo". Nhưng cách viết này
ghi giá trị vào `.data` của `nn.Parameter` rồi **đọc lại chính leaf parameter đó**.

### 6.2 Ba hậu quả

**(a) Knowledge mapper không bao giờ nhận gradient.** `.data` là view bỏ qua
autograd. Giá trị được sao chép sang, nhưng liên kết đồ thị bị cắt. Tensor trả
về là leaf parameter, nên backward dừng ngay tại đó.

Ví dụ tối thiểu — chạy để thấy:

```python
import torch, torch.nn as nn

lin = nn.Linear(4, 4)
p = nn.Parameter(torch.zeros(1, 4))

out = lin(p)
p.data.copy_(out[0])       # đúng cách bản gốc làm
loss = p.sum()             # rồi dùng chính parameter đó ở downstream
loss.backward()

print('grad của lin.weight:', lin.weight.grad)   # -> None
```

Áp vào mô hình thật: `photo2sketch_net`, `sketch2photo_net`,
`attn_pooling_photo_nets`, `attn_pooling_sketch_nets`, `photo_proxy_tokens`,
`sketch_proxy_tokens` — **31,916,544 tham số, tức 99.6% nhóm prompt** — có
`grad = None` vĩnh viễn. Chúng nằm trong optimizer, được đếm vào "trainable
params", nhưng đứng yên ở giá trị khởi tạo từ đầu tới cuối.

Đây là cơ chế *Hierarchical Cross-modal Prompt Learning* — đóng góp chính của
HiCroPL. Nó không hoạt động.

**(b) Param bị thay đổi ngoài `optimizer.step()`.** Mỗi forward ghi đè giá trị
prompt. Adam vẫn giữ `exp_avg` / `exp_avg_sq` tính trên chuỗi giá trị cũ, nên
moment được áp lên một tensor đã bị thay nghĩa. Update của step trước bị xoá ở
forward kế tiếp.

**(c) Rò rỉ sang giai đoạn đánh giá.** `extract_eval_features()` gọi
`visual_visual_learner()` cho **mỗi batch validation**. Weights bị ghi đè liên
tục trong lúc eval, nên kết quả validation phụ thuộc vào số batch đã chạy, và
checkpoint lưu giá trị sau khi đã bị mutate.

### 6.3 Cách sửa: luồng functional

```python
# TRƯỚC — ghi ngược vào Parameter
for i in range(self.cross_layer):
    self.cross_prompts_sketch[i].data.copy_(updated_sketch_prompts[i])
return (self.cross_prompts_photo[0], self.cross_prompts_sketch[0],
        [self.cross_prompts_photo[i] for i in range(1, ...)],
        [self.cross_prompts_sketch[i] for i in range(1, ...)])

# SAU — dùng list tensor cục bộ, Parameter không bị đụng
photo_out  = list(self.cross_prompts_photo)
sketch_out = list(self.cross_prompts_sketch)
...
for i in range(self.cross_layer):
    sketch_out[i] = updated_sketch_prompts[i]
...
return photo_out[0], sketch_out[0], photo_out[1:], sketch_out[1:]
```

**Kết quả số học không đổi.** Hai stage không giao index — P→S chỉ chạm
`[0, cross_layer)`, S→P chỉ chạm `[cross_layer, prompt_depth)` — nên đọc từ list
cho ra đúng giá trị như bản cũ. Khác biệt duy nhất là đồ thị tính toán được giữ
nguyên, nên gradient chảy được về mapper, và `nn.Parameter` chỉ đổi qua
`optimizer.step()`.

Sau khi sửa, `visual_visual_learner()` trở thành **hàm thuần**: gọi hai lần liên
tiếp cho ra kết quả giống hệt. Đây là bất biến được kiểm trong
`scripts/verify_training.py` (check `prompt_learner_pure`).

---

## 7. Rò rỉ E — Prompt learner giữ tham chiếu backbone

### 7.1 Vấn đề

```python
class CrossModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, clip_model_distill=None):
        ...
        self.token_embedding = clip_model.token_embedding
        self.clip_model = clip_model
        self.clip_model_distill = clip_model_distill or clip_model
        self.ZS_image_encoder = self.clip_model_distill.visual
```

`nn.Module.__setattr__` tự động **đăng ký** mọi giá trị kiểu `nn.Module` thành
submodule. Nên `prompt_learner.parameters()` trả về cả CLIP backbone lẫn teacher.
Mà `configure_optimizers` dùng chính lời gọi đó để dựng nhóm `prompt_lr`:

```python
add_unique_params(self.model.visual_visual_learner.parameters(), prompt_params, seen_ids)
```

Class này hiện không nằm trên đường chạy (`CustomCLIP` dùng
`VisualVisualPromptLearner` + `SimpleTextPromptLearner`), nhưng nếu ai đó bật
lại thì toàn bộ backbone sẽ vào optimizer với learning rate của prompt.

### 7.2 Cách sửa

Bọc trong `list` để `__setattr__` không nhận diện là module, rồi expose qua
property nên API bên ngoài giữ nguyên:

```python
self._clip_model = [clip_model]
self._clip_model_distill = [clip_model_distill or clip_model]
self._ZS_image_encoder = [self._clip_model_distill[0].visual]

@property
def clip_model(self):
    return self._clip_model[0]
```

---

## 8. Lớp kiểm chứng

Sửa xong chưa đủ — cần cơ chế phát hiện nếu lỗi tương tự quay lại.

### 8.1 Kiểm tra lúc chạy: `_assert_no_param_leak()`

Chạy ở `on_fit_start`, trước khi tốn bất kỳ epoch nào. Kiểm hai bất biến:

1. Không param nào của teacher (`clip_distill_*`) trainable.
2. Trong backbone student, **nguồn trainable hợp lệ duy nhất là LayerNorm**
   (cộng `logit_scale` nếu bật tường minh).

Điểm quan trọng về cách cài đặt: so bằng `id()` với tập LayerNorm thật, **không**
match theo tên:

```python
ln_ids = {
    id(p)
    for _, m in model.named_modules() if isinstance(m, torch.nn.LayerNorm)
    for p in m.parameters(recurse=False)
}
backbone = [n for n, p in trainable
            if not n.startswith(learner_prefixes)
            and not n.startswith(distill_prefixes)
            and id(p) not in ln_ids]
```

Loại trừ prompt learner theo tên là bắt buộc: chúng có `nn.MultiheadAttention`
riêng và `in_proj_weight` của chúng **đúng là phải trainable**.

### 8.2 Kiểm tra chủ động: `scripts/verify_training.py`

17 check trên **CLIP thật**, chia năm nhóm:

| Nhóm | Check |
|---|---|
| Freeze policy | `teacher_frozen`, `backbone_only_ln`, `bare_params_frozen`, `trainable_budget` |
| Optimizer | `optimizer_matches_trainable`, `optimizer_no_teacher`, `assert_hook_runs` |
| Mutation | `forward_no_mutation`, `prompt_learner_pure`, `eval_path_no_mutation` |
| Gradient | `grad_reaches_all_trainable`, `grad_reaches_knowledge_mapper`, `teacher_no_grad` |
| Khác | `step_updates_prompts_only`, `teacher_eval_mode`, `logit_scale_clamped`, `neg_branch_matches_loss` |

Script dựng **model thật** thay vì mock, có chủ đích: toàn bộ rò rỉ ở mục A bắt
nguồn từ cấu trúc module thật của CLIP (`nn.MultiheadAttention.in_proj_weight`
là bare Parameter), thứ mà mock không tái hiện được.

```bash
python scripts/verify_training.py            # cấu hình mặc định
python scripts/verify_training.py --all      # thêm 6 cấu hình ablation
python scripts/verify_training.py --list     # xem danh sách (không cần torch)
```

Exit code 0/1 nên dùng được trong CI.

Hai check bắt trực tiếp rò rỉ D:

- `prompt_learner_pure` — gọi learner hai lần liên tiếp phải ra cùng kết quả.
  Bản `.data.copy_()` fail vì param bị ghi đè mỗi forward.
- `grad_reaches_knowledge_mapper` — sáu nhóm param của cross-modal flow phải có
  gradient khác 0.

---

## 9. Trước và sau

### 9.1 Log khởi động

```
# TRƯỚC
clip_photo:          trainable 31,516,417 / 151,277,313
clip_sketch:         trainable 31,516,417 / 151,277,313
clip_distill_photo:  trainable 31,516,417 / 151,277,313
clip_distill_sketch: trainable 31,516,417 / 151,277,313
Number of trainable prompt params:      31,937,024   ← nhưng grad = None
Number of trainable non-prompt params: 126,065,668   ← 4×31,450,881 rò rỉ + 4×65,536 LN
TỔNG: 158,002,692

# SAU
clip_photo:          trainable 65,536 / 151,277,313
clip_sketch:         trainable 65,536 / 151,277,313
clip_distill_photo:  trainable 0      / 151,277,313
clip_distill_sketch: trainable 0      / 151,277,313
Number of trainable prompt params:  31,937,024   ← giờ thật sự nhận gradient
Number of trainable non-prompt params: 131,072
TRAINABLE: 308 tensors, 32,068,096 params
```

### 9.2 Điều gì thay đổi về mặt huấn luyện

| Khía cạnh | Trước | Sau |
|---|---|---|
| Backbone CLIP | đang fine-tune 126M params | đóng băng thật |
| LayerNorm student | có train (lẫn trong 126M) | có train, chỉ 131,072 |
| Teacher | trôi theo student | hằng số |
| Knowledge mapper | grad = None, đứng yên ở init | học bình thường |
| Prompt params | bị ghi đè mỗi forward | chỉ đổi qua `optimizer.step()` |
| Validation | mutate weights mỗi batch | chỉ đọc |

**Lưu ý khi so sánh kết quả:** đây không phải "cùng mô hình chạy lại". Trước khi
sửa, cái đang được huấn luyện là *CLIP fine-tuning một phần* với prompt cố định
ở giá trị ngẫu nhiên. Sau khi sửa mới đúng là *prompt tuning*. Mọi số đo trước
đó không dùng để so sánh trực tiếp được.

---

## 10. Tự kiểm tra

### 10.1 Nhanh nhất — đọc log khởi động

```
clip_photo: trainable 65,536 / total 151,277,313 params
```

65,536 là con số kiểm chứng được bằng tay (mục 2). Lệch khỏi số này nghĩa là
freeze policy đã hỏng.

### 10.2 Đầy đủ

```bash
python scripts/verify_training.py --all
```

### 10.3 Kiểm gradient thủ công

Sau `loss.backward()` của step đầu tiên:

```python
for n, p in model.model.visual_visual_learner.named_parameters():
    assert p.grad is not None, f"{n} không nhận gradient"
```

Bản trước khi sửa fail assert này ở `photo2sketch_net.*`, `sketch2photo_net.*`,
`attn_pooling_*`, `*_proxy_tokens`.

---

## 11. Ghi chú: 32M tham số trainable có hợp lý không?

Sau khi sửa, `prompt params = 31,937,024` — trong đó chỉ **20,480 là prompt token
thật**, còn 31,916,544 là mạng knowledge mapper và LKP. Chúng lớn vì mỗi
`CrossPromptAttention` là một khối transformer đầy đủ ở 768 chiều (MHA 2.36M +
ba lớp `linear_q/k/v` 1.77M + FFN mở rộng 4× chiếm 4.72M), và `_get_clones` tạo
`prompt_depth` bản `AttentionPooling` độc lập không chia sẻ weight.

Kích thước này **không phụ thuộc `n_ctx`** — giảm `n_ctx` từ 4 xuống 2 chỉ tiết
kiệm vài nghìn tham số, vì `n_ctx` là độ dài chuỗi còn tham số tỉ lệ với
`d_model²`.

Đây là đặc tính thiết kế của HiCroPL, không phải rò rỉ — nhưng đáng lưu ý khi
báo cáo con số "trainable params", và đáng đo bằng ablation
`--disable_cross_exchange` (bỏ toàn bộ mapper, còn 151,552 params).

---

## 12. Tham chiếu

| Mã | File | Nội dung |
|---|---|---|
| A | `src/model_hicropl.py` | `freeze_model` + `unfreeze_ln`, `freeze_all_but_bn` DEPRECATED |
| B | `src/model_hicropl.py` | `CustomCLIP.train()`, `torch.no_grad()`, `_encode_gpt_distill()` |
| C | `src/model_hicropl.py` | `configure_optimizers` whitelist + fail-fast |
| D | `src/hicropl.py` | `VisualVisualPromptLearner.forward`, `CrossModalPromptLearner.forward` |
| E | `src/hicropl.py` | `CrossModalPromptLearner.__init__` list-wrapper + property |
| — | `src/model_hicropl.py` | `_assert_no_param_leak()` |
| — | `scripts/verify_training.py` | 17 check |
| — | `tests/test_hicropl.py` | `test_knowledge_mapper_receives_gradient`, `test_bidirectional_flow_modifies_prompts` |
