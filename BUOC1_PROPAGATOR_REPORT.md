# Báo cáo Bước 1 — "Làm propagator của riêng mình, khác HiCroPL"

Theo đúng thứ tự checklist. Trạng thái: ✅ xong có số liệu | ⚠️ xong nhưng có vấn đề cần quyết định | ❌ chưa làm (thiếu code hoặc thiếu run).

---

## ☑ Sketch-queried extraction

**Trạng thái: ✅ đã cài, đã chạy, đúng công thức `q_l = W_q · mean(Z_l)`.**

Cài trong `VisualVisualPromptLearner` (`src/hicropl.py`), cờ `--propagator_sketch_queried --propagator_gate_mode none`, module hoàn toàn tách biệt khỏi LKP/Mapper cũ (không share instance với `attn_pooling_photo_nets`/`photo2sketch_net`).

```
q_l   = self.q_proj[l](Z[l].mean(0, keepdim=True))
src_l = Pp[l].detach()                                # photo read-only, detach tại INPUT
r_l   = self.pool[l](q_l, src_l, src_l)
R, Q  = torch.cat(r, 0), torch.cat(Z, 0)
```

Kết quả (Sketchy-Ext-2, `n_ctx=3, prompt_depth=12, cross_layer=12`, 15 epoch): **mAP@200 = 0.7950, P@200 = 0.7684**, best_epoch=2.

---

## ☑ Gate theo độ sâu, khởi tạo bằng 0

**Trạng thái: ⚠️ đã cài đúng công thức, đã verify identity tại bước 0 tuyệt đối, nhưng phát hiện một vấn đề hội tụ nghiêm trọng.**

```
delta = self.mapper_update.forward_delta(Q, R, R)     # không có residual của query
gate  = torch.tanh(self.gamma).repeat_interleave(m)[:, None]
P_hat = Q + gate * delta                                # gamma init 0
```

**Verify độc lập (script `scripts/check_propagator_identity.py`, chạy TRƯỚC training thật):**
- `torch.allclose(P_hat, Q)` đúng tuyệt đối tại bước 0 (`cos = 1.000000` mọi layer) — xác nhận công thức zero-init đúng.
- `cross_prompts_photo` không nhận gradient — photo read-only đúng như thiết kế.
- `q_proj`/`pool` **không** nhận gradient tại bước 0 — **đây là bắt buộc về toán học** (chứng minh bằng cách đo từng tham số con của `mapper_update`: chỉ `attn.out_proj`/`ffn.c_proj` — hai lớp bị zero-init — nhận gradient ở bước đầu; mọi thứ phía sau bị chặn bởi ma trận-0), không phải bug. Sau khi nhích γ ra khỏi 0 thủ công, gradient chảy lại bình thường vào `q_proj`/`pool` — xác nhận cơ chế thoát khỏi điểm 0 hoạt động đúng.

**Vấn đề phát hiện khi train thật (15 epoch):** `gate_mode=learned, rank=0` cho `best_epoch=0` — tức điểm tốt nhất rơi ngay validation đầu tiên, **tệ dần** suốt 14 epoch còn lại (`mAP@200 = 0.7942, P@200 = 0.7686` tại điểm tốt nhất). Khớp với chẩn đoán trước đó: gradient tại bước đầu đẩy γ về phía 0 vì `delta` (từ module còn gần ngẫu nhiên) gây hại nhiều hơn lợi trong giai đoạn đầu. **Hiện tượng này không biến mất khi tăng từ 10 lên 15 epoch** — cho thấy đây là vấn đề bản chất cơ chế ở ngân sách epoch hiện có, không phải thiếu thời gian.

**Cần quyết định trước khi đi tiếp:** có chấp nhận `learned` gate như hiện tại (rủi ro cao), hay ưu tiên `none`/`zero_init`?

---

## ☑ Tiêu chí chấp nhận trên Sketchy-Ext-2 (mAP@200 ≥ 80.2)

**Trạng thái: ❌ CHƯA ĐẠT ở mọi cấu hình đã thử.**

| Cấu hình | mAP@200 | Đạt ≥80.2? |
|---|---|---|
| `gate=none, rank=0` | 0.7950 | ✗ |
| `gate=none, rank=32` | 0.7909 | ✗ |
| `gate=none, rank=64` | **0.8019** | ✗ (thiếu 0.01) |
| `gate=zero_init, rank=0` | 0.7956 | ✗ |
| `gate=zero_init, rank=32` | 0.8000 | ✗ |
| `gate=zero_init, rank=64` | 0.7977 | ✗ |
| `gate=learned, rank=0` | 0.7942 | ✗ |
| `gate=learned, rank=32` | 0.7965 | ✗ |
| `gate=learned, rank=64` | 0.7938 | ✗ |
| `--exchange_sketch_retrieval` (biến thể khác, 10 epoch) | **0.8009** | ✗ (thiếu, nhưng ít epoch nhất) |

→ Theo đúng checklist: **chuyển sang phương án dự phòng** ở bước tiếp theo.

---

## ☑ Phương án dự phòng: low-rank bottleneck

**Trạng thái: ⚠️ đã cài, đã chạy `r∈{32,64}`, CHƯA chạy `r∈{16,128}`.**

```
R = W_up(W_down(R))    # W_down: 768→r, W_up: r→768, không bias
```

Đã verify: `r=0` bit-exact với không có bottleneck; `r=32` cho đúng `2×768×32=49,152` tham số chênh lệch, gradient chảy vào cả `W_down`/`W_up`.

**Không có xu hướng đơn điệu theo rank** (xem ma trận đầy đủ trong `REPORT_propagator_ablation_sketchy2.md`):
- `gate=none`: r=32 làm tệ hơn, r=64 làm tốt hơn (tốt nhất toàn bảng: 0.8019).
- `gate=zero_init`: cả r=32 và r=64 đều tốt hơn r=0, r=32 tốt nhất trong nhóm này.
- `gate=learned`: cả hai rank chỉ nhích nhẹ, không rõ hướng.

Với 1 seed, không đủ căn cứ chọn rank tối ưu. **Cần chạy `r=16` và `r=128`** để có đủ lưới trước khi kết luận (theo đúng danh sách checklist), và **cần multi-seed** trước khi báo cáo số cuối.

---

## ☑ Ablation toán tử truyền tri thức (bảng chính)

**Trạng thái: ⚠️ điền được 3/6 hàng bằng dữ liệu thật, 1 hàng thiếu dữ liệu ở đúng cấu hình, 1 hàng thiếu code, 1 hàng cần làm rõ ánh xạ.**

| Cấu hình | #params | mAP@200 | P@200 | Nguồn |
|---|---|---|---|---|
| Sketch prompt độc lập | — | — | — | ❌ chưa có run ở đúng `n_ctx=3/depth=12/cross_layer=12/aug_shared_encoder`, 15 epoch — chỉ có `--disable_exchange --cross_layer 6` ở phiên bản kiểm hồi quy (không cùng cấu hình, không dùng được) |
| Linear projection theo layer (MaPLe/CLIP-AT) | — | — | — | ❌ **chưa có code** — cần cài mới, không có class nào trong `src/hicropl.py` làm việc này |
| HiCroPL proxy + mapper nguyên bản | — | — | — | ❌ chưa có run thật (chỉ có số tham số từ smoke-test cũ, không phải kết quả train) |
| HiCroPL + stop-gradient (PUTEA hiện tại) | 9,082,880 | 0.8042* | 0.7797* | ⚠️ *số này đo ở **10 epoch**, không phải 15 — lệch ngân sách với các hàng dưới, cần chạy lại ở 15 epoch để so công bằng |
| + sketch-queried extraction | 44,555,264† | 0.7950 | 0.7684 | ✅ (†suy từ cấu trúc — `q_proj`+`pool`+`mapper_update` giống hệt kích thước đã đo trực tiếp ở `--exchange_query_from_sketch`, chưa log trực tiếp dòng `PARAM_FP` cho chính propagator) |
| + gate theo độ sâu (mô hình đầy đủ) | 44,555,276† | 0.7942 | 0.7686 | ✅ nhưng xem cảnh báo `best_epoch=0` ở trên |

**2 việc cần làm trước khi bảng này hoàn chỉnh:**
1. Chạy `--disable_exchange` và mặc định-không-cờ ở đúng `n_ctx=3, prompt_depth=12, cross_layer=12, --aug_shared_encoder`, 15 epoch.
2. Cài "Linear projection theo từng layer" — thiết kế đơn giản nhất: 1 `nn.Linear(768,768)` cố định mỗi layer, không attention/pooling, chiếu trực tiếp `cross_prompts_photo[l]` (hoặc trung bình) vào `cross_prompts_sketch[l]`, không có LKP.
3. Chạy lại "HiCroPL + stop-gradient" ở đúng 15 epoch cho công bằng với các hàng propagator.

---

## ☐ Bảng hướng truyền (Table IX) với propagator mới

**Trạng thái: ❌ chưa làm được.**

`--propagator_sketch_queried` hiện **chỉ cài cho trường hợp `cross_layer == prompt_depth`** (toàn bộ là Photo→Sketch, không có Sketch→Photo). Muốn tái tạo Table IX (P→S, S→P, hai chiều) với propagator mới, cần mở rộng `forward()` để propagator cũng xử lý được dải `[cross_layer, prompt_depth)` theo chiều Sketch→Photo — hiện chưa có.

---

## ☐ Lưu dữ liệu để vẽ hình

**Trạng thái: ⚠️ một phần.** Có log `GATE_FP | tanh(gamma)_mean | tanh(gamma)_max` mỗi epoch (console, chưa dump ra file có thể vẽ lại theo từng layer riêng biệt — hiện chỉ có mean/max gộp cả `L` layer, không phải giá trị từng layer). Attention map trung bình theo head — **chưa có code trích xuất**, `nn.MultiheadAttention` mặc định không giữ lại attention weights khi `need_weights=False` (đang dùng trong toàn bộ codebase).

---

## ☐ Đổi tên "LKP"/"Knowledge Mapper" trong code

**Trạng thái: ❌ cố ý chưa làm**, đúng theo kế hoạch gốc: hoãn đổi tên tới khi có kết quả G0/G1/G2 để không đổi tên và đổi hành vi cùng lúc. G0/G1/G2 đã có kết quả (phần trên) — **có thể làm bước này bây giờ nếu muốn**, nhưng nên đợi quyết định xong nhánh nào (none/zero_init/learned, rank nào) sẽ là kiến trúc cuối cùng trước khi đổi tên, tránh đổi tên 2 lần.

---

## ☐ Kiểm tra inference và chi phí

**Trạng thái: ❌ chưa đo.** Cần, sau khi chốt cấu hình cuối cùng: (1) tính `P̂` một lần, cache, so kết quả truy hồi với không cache — script tương tự `train==eval` đã dùng để bắt bug trước đây, có thể tái dùng logic; (2) đo GFLOPs lúc inference; (3) đo thời gian mỗi epoch (đã có trong log Kaggle, `elapsed` ~1h13' cho mọi run propagator ở 15 epoch — khá đồng đều, không có cấu hình nào đột biến chi phí).

---

## Tóm tắt việc cần làm tiếp, theo ưu tiên

1. **Quyết định hướng gate** (`none` vs `zero_init` vs `learned`) — dữ liệu hiện tại nghiêng về `none`/`zero_init`, `learned` có rủi ro hội tụ nghiêm trọng (`best_epoch=0`).
2. Chạy 3 hàng còn thiếu số liệu thật trong bảng ablation chính (disable_exchange, mặc định không cờ, và PUTEA-hiện-tại-ở-15-epoch).
3. Cài "Linear projection theo layer" (~30 phút code) để hoàn thiện bảng.
4. Chạy nốt `bottleneck rank∈{16,128}`.
5. Chỉ sau khi 1-4 xong: multi-seed cho cấu hình đã chọn, rồi mới đổi tên code + đo inference/GFLOPs (2 việc cuối, không tốn thời gian GPU, làm sau cùng).
