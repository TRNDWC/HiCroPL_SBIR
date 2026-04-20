# Kế hoạch implement Cross-Modal Prompt Sketch-Photo (Cách 1)

## 1) Bối cảnh hiện tại (Current State)

Hệ thống hiện tại trong nhánh HiCroPL đang hoạt động như sau:

- Có 2 prompt learner độc lập:
  - `prompt_learner_photo`
  - `prompt_learner_sketch`
- Mỗi learner trả về bộ output (1):
  - `text_input`, `tokenized_prompts`, `fixed_embeddings`
  - `cross_prompts_text_deeper`
  - `cross_prompts_visual_deeper`
- Mỗi nhánh encoder (`_forward_branch`) dùng trực tiếp prompt từ learner tương ứng.
- Text encoder và visual encoder của mỗi modality (photo/sketch) đang chạy độc lập theo prompt của chính modality đó.
- Nhánh negative image (`neg_tensor`) đang đi qua nhánh photo.

### Định nghĩa Early/Deeper hiện tại theo `cross_layer`

Giữ nguyên đúng định nghĩa hiện tại trong code:

- Early layers: chỉ số `i` thuộc `[0, cross_layer - 1]`
- Deeper layers: chỉ số `i` thuộc `[cross_layer, prompt_depth - 1]`

Ý nghĩa trong learner hiện tại:

- Ở Early: dùng text proxy (LKP trên text prompts) để cập nhật visual prompts (T -> I).
- Ở Deeper: dùng visual proxy (LKP trên visual prompts) để cập nhật text prompts (I -> T).

## 2) Mục tiêu mới (Target Design)

Bạn muốn bổ sung thêm một tầng cập nhật prompt giữa 2 modality sketch và photo:

- Tạo class mới `CrossModalPromptLearner_SketchPhoto`.
- Class này kế thừa cơ chế cập nhật prompt chéo phân tầng kiểu HiCroPL, nhưng chỉ áp dụng cho tương tác Sketch <-> Photo.
- Chỉ cập nhật **visual prompts**.

### Luồng mới mong muốn

- Bước A: Lấy output (1) từ 2 learner gốc (`photo` và `sketch`).
- Bước B: Đưa visual prompts của sketch/photo vào `CrossModalPromptLearner_SketchPhoto` để tạo output (2).
- Bước C: Khi encode:
  - Text prompts: dùng từ output (1) (giữ nguyên)
  - Visual prompts: dùng từ output (2)

## 3) Chốt phương án Cách 1

Áp dụng đúng quyết định của bạn:

- Output (2) được tính **một lần cho mỗi batch**.
- Bộ visual prompts photo sau cập nhật được dùng chung cho:
  - photo dương
  - photo âm (`neg_tensor`)

Lợi ích:

- Giảm chi phí tính toán
- Giữ prompt nhất quán trong cùng batch
- Tránh lệch hành vi giữa positive/negative trong cùng photo branch

## 4) Thiết kế module mới

## 4.1 Class mới: `CrossModalPromptLearner_SketchPhoto`

Vị trí đề xuất:

- `src/hicropl.py`

Input:

- `photo_visual_prompts`: list/tensor visual prompts của photo từ output (1)
- `sketch_visual_prompts`: list/tensor visual prompts của sketch từ output (1)

Output (2):

- `photo_visual_prompts_updated`
- `sketch_visual_prompts_updated`

Ràng buộc:

- Không thay đổi text prompts.
- Early/Deeper phải tách đúng theo `cross_layer` như phần 1.

## 4.2 Cơ chế cập nhật bên trong class mới

Theo đúng tinh thần phân tầng:

- Early (`0 -> cross_layer-1`):
  - Nén sketch prompts bằng LKP -> sketch proxy
  - Dùng CrossPromptAttention cập nhật photo visual prompts
- Deeper (`cross_layer -> prompt_depth-1`):
  - Nén photo prompts bằng LKP -> photo proxy
  - Dùng CrossPromptAttention cập nhật sketch visual prompts

Ghi chú quan trọng:

- Chỉ thao tác trên visual prompt dimension (`v_dim = 768` với ViT-B/32 theo setup hiện tại).
- Duy trì số prompt tokens là `n_ctx` cho mỗi layer.

## 5) Thay đổi trong `CustomCLIP`

Vị trí chính:

- `src/model_hicropl.py`

## 5.1 Khởi tạo module mới

Trong `CustomCLIP.__init__`:

- Khởi tạo `self.prompt_learner_sketch_photo = CrossModalPromptLearner_SketchPhoto(...)`
- Truyền `prompt_depth`, `cross_layer`, `n_ctx`, `v_dim` theo config hiện có.

## 5.2 Luồng forward mới

Trong `CustomCLIP.forward`:

- Gọi 2 learner gốc để lấy output (1) cho photo/sketch.
- Trích visual prompts từ output (1) của cả 2 bên.
- Gọi module sketch-photo để nhận output (2) (chỉ một lần mỗi batch).
- Encode:
  - Text encoder photo/sketch dùng text prompts từ output (1)
  - Visual encoder photo/sketch dùng visual prompts từ output (2)
  - Nhánh `neg_tensor` dùng lại visual prompts photo từ output (2)

## 5.3 Điều chỉnh `_forward_branch`

Mục tiêu:

- Cho phép truyền visual prompts từ ngoài vào `_forward_branch`.

Đề xuất:

- Thêm tham số optional:
  - `first_visual_prompt_override`
  - `deeper_visual_prompts_override`
- Nếu có override thì dùng override; không thì fallback về prompt learner gốc (để tương thích ngược).

## 6) Optimizer và trainable params

Vị trí:

- `HiCroPL_SBIR.configure_optimizers` trong `src/model_hicropl.py`

Yêu cầu:

- Thêm tham số của `prompt_learner_sketch_photo` vào nhóm `prompt_params` (LR = `prompt_lr`).
- Không đổi cơ chế train LayerNorm hiện tại.

## 7) Kiểm thử và xác minh

## 7.1 Kiểm thử shape và mapping layer

Checklist:

- Số layer visual prompt trước/sau update giữ nguyên `prompt_depth`.
- Early split đúng `[0, cross_layer-1]`.
- Deeper split đúng `[cross_layer, prompt_depth-1]`.
- Shape mỗi prompt layer giữ `[n_ctx, v_dim]`.

## 7.2 Kiểm thử forward train

- Chạy smoke test 1 batch:
  - không lỗi shape
  - không lỗi dtype/device
  - loss vẫn tính được

## 7.3 Kiểm thử nhánh neg

- Xác nhận nhánh `neg_tensor` dùng visual prompts photo từ output (2), không tính lại output (2).

## 8) Kế hoạch triển khai theo bước

1. Tạo class `CrossModalPromptLearner_SketchPhoto` trong `src/hicropl.py`.
2. Tích hợp class mới vào `CustomCLIP.__init__`.
3. Refactor nhẹ `_forward_branch` để nhận visual prompt override.
4. Cập nhật `CustomCLIP.forward` theo luồng output (1) -> output (2) -> encode.
5. Cập nhật optimizer để include params module mới.
6. Thêm/điều chỉnh test tối thiểu cho shape + flow.
7. Chạy smoke test train step và sửa lỗi phát sinh.

## 9) Tiêu chí hoàn thành (Definition of Done)

- Build/Run forward train không lỗi.
- Cơ chế output (1)/output (2) đúng theo mô tả mới.
- Text prompts dùng từ output (1), visual prompts dùng từ output (2).
- Early/Deeper đúng định nghĩa hiện tại theo `cross_layer`.
- `neg_tensor` dùng chung output (2) theo Cách 1.
- Params class mới được optimizer cập nhật.

## 10) Ghi chú rủi ro kỹ thuật

- Cần tránh cập nhật prompt bằng thao tác phá graph autograd trong phần mới.
- Cần giữ nhất quán dtype (`fp32/fp16`) với flow hiện tại của CLIP/HiCroPL.
- Nếu có cache prompt/classnames, cần đảm bảo cache không làm stale output (2) theo batch.
