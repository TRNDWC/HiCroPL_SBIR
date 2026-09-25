# Báo cáo thực nghiệm — Propagator mới trên Sketchy-Ext-2

10 lần chạy, `n_ctx=3, prompt_depth=12, cross_layer=12, --aug_shared_encoder`, ngày 2026-09-24. Không có seed cố định trong dữ liệu gửi lên — mọi kết luận dưới đây coi là **1 seed duy nhất**, chưa đủ độ tin cậy theo đúng quy tắc checklist ("mức cải thiện nhỏ hơn độ lệch chuẩn giữa các seed thì không tính").

## ⚠️ Cảnh báo trước khi đọc bảng: `exp_name` không khớp cờ thật

4 dòng có hậu tố `_none`/`_learned` trong tên nhưng **không phải `--propagator_gate_mode zero_init`** như phần đầu tên gợi ý — đây là lỗi đặt tên, không phải lỗi cấu hình. Tôi đã tra lại đúng cờ thật từ cột `command`:

| `exp_name` (gây hiểu nhầm) | `--propagator_gate_mode` THẬT | `--propagator_bottleneck_rank` THẬT |
|---|---|---|
| `..._r32_none` | `none` | 32 |
| `..._r64_none` | `none` | 64 |
| `..._r32_learned` | `learned` | 32 |
| `..._r64_learned` | `learned` | 64 |
| `..._r32` (không hậu tố) | `zero_init` | 32 |
| `..._r64` (không hậu tố) | `zero_init` | 64 |

Bảng dưới đây dùng **cờ thật**, không dùng tên.

## Bảng đầy đủ

| Cơ chế | `gate_mode` | `rank` | epochs | best_epoch | mAP@200 | P@200 |
|---|---|---|---|---|---|---|
| `--exchange_sketch_retrieval` | — | — | 10 | 6 | **0.8009** | **0.7745** |
| `--propagator_sketch_queried` | none | 0 | 15 | 2 | 0.7950 | 0.7684 |
| `--propagator_sketch_queried` | none | 32 | 15 | 11 | 0.7909 | 0.7635 |
| `--propagator_sketch_queried` | none | 64 | 15 | 3 | **0.8019** | **0.7746** |
| `--propagator_sketch_queried` | zero_init | 0 | 15 | 7 | 0.7956 | 0.7693 |
| `--propagator_sketch_queried` | zero_init | 32 | 15 | 11 | 0.8000 | 0.7711 |
| `--propagator_sketch_queried` | zero_init | 64 | 15 | 8 | 0.7977 | 0.7691 |
| `--propagator_sketch_queried` | learned | 0 | 15 | **0** ⚠️ | 0.7942 | 0.7686 |
| `--propagator_sketch_queried` | learned | 32 | 15 | 2 | 0.7965 | 0.7681 |
| `--propagator_sketch_queried` | learned | 64 | 15 | 1 | 0.7938 | 0.7688 |

## Ma trận gate_mode × rank (mAP@200 / P@200)

| gate_mode \\ rank | 0 | 32 | 64 |
|---|---|---|---|
| **none** | 0.7950 / 0.7684 | 0.7909 / 0.7635 | **0.8019 / 0.7746** |
| **zero_init** | 0.7956 / 0.7693 | 0.8000 / 0.7711 | 0.7977 / 0.7691 |
| **learned** | 0.7942 / 0.7686 | 0.7965 / 0.7681 | 0.7938 / 0.7688 |

## Phát hiện chính

**1. Không cấu hình nào đạt tiêu chí chấp nhận `mAP@200 ≥ 80.2`.** Cao nhất là `none, rank=64` với `0.8019` — thiếu đúng `0.01`, trong biên độ nhiễu 1-seed, nhưng vẫn chưa đạt theo đúng ngưỡng đã đặt ra.

**2. `--exchange_sketch_retrieval` (cơ chế khác hẳn propagator, không gate, không bottleneck) đang là kết quả tốt nhất trong toàn bộ 10 dòng — và chỉ chạy 10 epoch, ít hơn 5 epoch so với 9 dòng còn lại.** Đáng để đầu tư thêm epoch cho cơ chế này trước khi kết luận propagator mới là hướng đúng.

**3. Cả 2 cơ chế đều CHƯA vượt được baseline cũ `--exchange_detach_source` đơn thuần (`80.42/77.97` ở 10 epoch, đã ghi nhận trước đó).** Dù propagator chạy nhiều epoch hơn (15 vs 10), không cấu hình nào trong bảng trên vượt được con số đó. Đây là phát hiện quan trọng nhất, cần báo cáo trung thực: **tính đến thời điểm này, chưa có bằng chứng propagator mới tốt hơn cơ chế cũ trên Sketchy-Ext-2.**

**4. `gate_mode=learned, rank=0` (chính là **G2**, mô hình "đầy đủ" theo thiết kế gốc) có `best_epoch=0`.** Nghĩa là điểm tốt nhất rơi ngay ở lần validate đầu tiên, rồi **tệ dần** suốt 14 epoch còn lại. Đây khớp chính xác với hiện tượng đã chẩn đoán trước đó ở ngân sách 10 epoch (γ bị gradient đẩy về 0 do delta ban đầu mang nhiều nhiễu hơn lợi ích) — hiện tượng này **không biến mất** dù tăng lên 15 epoch, cho thấy vấn đề không phải do thiếu thời gian train mà do bản chất cơ chế.

**5. Bottleneck không cho một xu hướng nhất quán:**
- `none`: rank=32 làm **tệ hơn** (0.7950→0.7909), rank=64 làm **tốt hơn** (0.7950→0.8019).
- `zero_init`: rank=32 tốt hơn (0.7956→0.8000), rank=64 cũng tốt hơn nhưng ít hơn (→0.7977).
- `learned`: cả rank=32 và rank=64 đều chỉ nhích nhẹ, không rõ xu hướng.

Không có mẫu hình đơn điệu theo rank ở bất kỳ gate_mode nào — với 1 seed, nhiều khả năng đây là nhiễu, không phải tín hiệu thật.

**6. `best_epoch` dao động rất lớn giữa các cấu hình (0, 1, 2, 3, 6, 7, 8, 11)** — dấu hiệu kết quả không ổn định giữa các lần chạy, càng củng cố lý do cần multi-seed trước khi đưa số liệu vào bài.

## Đề xuất bước tiếp theo

1. **Ưu tiên `--exchange_sketch_retrieval`** — chạy thêm epoch (15, giống các dòng khác) để so công bằng, vì đây là ứng viên mạnh nhất hiện tại và chưa được cho đủ ngân sách.
2. **Điều tra riêng `gate_mode=learned, rank=0`** — log giá trị `tanh(γ_l)` qua từng epoch (đã có `GATE_FP` sẵn trong log) để xác nhận có đúng là γ bị đẩy về 0 hay không, trước khi quyết định bỏ hướng "gate học được".
3. **Không kết luận gì về bottleneck rank** cho tới khi có ≥3 seed cho mỗi ô trong ma trận — dữ liệu hiện tại không đủ để chọn rank tối ưu.
4. **Chạy lại `--exchange_detach_source` đơn thuần ở đúng 15 epoch** để có đối chứng công bằng với toàn bộ 9 dòng propagator (hiện đang so với con số 10-epoch, không hoàn toàn công bằng dù kết luận #3 ở trên nhiều khả năng vẫn đúng).
