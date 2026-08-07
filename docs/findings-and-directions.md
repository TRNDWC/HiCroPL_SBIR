# Tổng kết thực nghiệm và hướng phát triển

Tài liệu này tổng hợp toàn bộ kết quả đo được trên HiCroPL-SBIR, chẩn đoán rút
ra từ chúng, và các hướng cải thiện — **mỗi hướng ghi rõ dựa trên bằng chứng
nào**.

Quy ước xuyên suốt, để phân biệt mức độ chắc chắn:

| Ký hiệu | Nghĩa |
|---|---|
| **[ĐO]** | Số liệu trực tiếp từ thực nghiệm |
| **[SUY]** | Suy ra từ [ĐO] bằng lập luận, chưa đo trực tiếp |
| **[GIẢ]** | Giả thuyết chưa kiểm chứng |

Cấu hình tham chiếu: CLIP ViT-B/32, Sketchy-ext, `n_ctx=2, prompt_depth=12,
language_depth=1, cross_layer=6, prompt_lr=1e-4, clip_LN_lr=1e-6,
batch_size=128, disable_augmentation, eval_mode=category`.

---

## 1. Tóm tắt

**Chẩn đoán:** mô hình đạt đỉnh ở epoch 2–3 rồi suy giảm, trong khi train loss
vẫn giảm. Ba mũi can thiệp độc lập đều thất bại theo cách loại trừ lẫn nhau, dẫn
tới kết luận: **nút thắt là tổng quát hoá, không phải capacity, và không phải
một khuyết tật của thành phần nào.**

| Câu hỏi | Kết quả | Trạng thái |
|---|---|---|
| Cross exchange (46.1M params) có giúp không? | −0.17 pp, dưới δ_min | **Trung tính** |
| L4 (cross-entropy) có gây overfit? | Bỏ đi → **tệ hơn 0.83 pp** | **Bác bỏ** |
| Weight decay có chặn được suy giảm? | Đổi best **0.014 pp** | **Bác bỏ** |
| Capacity lớn hơn có giúp? | `n_ctx` 2→16: suy giảm tăng **9.8×** | **Bác bỏ, ngược dấu** |
| Cho mô hình tự chọn tỉ lệ trộn residual? | α → 1, mAP **tệ hơn 0.51–0.80 pp** | **Bác bỏ, ngược dấu** |

**Mẫu hình xuyên suốt: mọi can thiệp làm tăng mức độ thích nghi đều làm hiệu
năng tệ đi; mọi can thiệp làm giảm nó thì trơ hoặc cũng tệ đi.** Điểm vận hành
hiện tại nằm ở phía "thích nghi quá nhiều" của đường cong.

Hệ quả: hướng đi phải là làm cho **bản thân quá trình thích nghi bảo toàn tính
tổng quát hơn**, chứ không phải điều chỉnh cường độ của nó.

> **Cảnh báo phương pháp (§3.7):** `ValidDataset` dùng đúng `UNSEEN_CLASSES`, tức
> **không có tập validation riêng** — best epoch đang được chọn trên chính tập
> test. Điều này không làm hỏng các so sánh trong tài liệu (mọi cấu hình đều chịu
> cùng một thiên lệch) nhưng khiến **con số tuyệt đối 78.4 là lạc quan**.

---

## 2. Phương pháp đo

Phần này quan trọng vì mọi kết luận phía sau đều phụ thuộc vào nó. Chênh lệch
giữa các cấu hình ở bài toán này nhỏ hơn nhiễu seed, nên nếu không thiết lập
thang đo trước thì mọi thí nghiệm đều là đuổi theo nhiễu.

### 2.1 δ_min — ngưỡng phân giải

**[ĐO]** Ba seed cùng cấu hình cho mAP@200 = 78.50 / 78.60 / 78.00, tức
**std = 0.32 pp**. Kiểm định cặp giữa hai seed cho KTC bootstrap 95%
`[−0.309, +0.112]` pp.

> **δ_min ≈ 0.31 pp.** Mọi chênh lệch mAP nhỏ hơn mức này là không kết luận được
> với dữ liệu một seed.

**[ĐO]** Huấn luyện **tất định theo seed**: ba lần chạy cùng seed 1 cho mAP
giống hệt tới 4 chữ số (0.7850). Nên toàn bộ dao động quan sát được là dao động
**seed**, không phải nhiễu chạy lại.

### 2.2 Kiểm định cặp thay vì so hai trung bình

Hai mô hình được đánh giá trên **cùng 12,694 query, cùng thứ tự**, và tương quan
rất cao (r ≈ 0.92–0.98) vì dùng chung backbone đóng băng và residual mix. So
theo cặp từng query khử được phần phương sai chung đó.

**[ĐO]** Trên phép so thật ở Giai đoạn 2b: SE không cặp 0.363 pp, SE có cặp
0.056 pp — **nhạy hơn 6.5 lần**.

Công cụ: `scripts/paired_test.py`, đọc `<run_dir>/ap_best.npz` do quá trình
train ghi ở epoch tốt nhất.

### 2.3 SUY GIẢM là chỉ số nhạy hơn mAP

Suy giảm = `best − last` đo **trong một run**, nên dao động seed dịch chuyển cả
hai số cùng chiều và bị khử.

**[ĐO]** Trên 3 seed Giai đoạn 0: best có std 0.32 pp, suy giảm có std
**0.12 pp** — nhạy hơn **2.7 lần** cho đúng câu hỏi "cấu hình nào overfit ít
hơn". Nhờ vậy 1 seed đủ để sàng lọc.

Công cụ: `scripts/summarize_runs.py --degradation`.

---

## 3. Kết quả

### 3.1 Hiện tượng nền: overfit sau 3 epoch

**[ĐO]** Ba seed, 10 epoch:

| Seed | best | @epoch | last (ep9) | suy giảm |
|---|---:|---:|---:|---:|
| 1 | 78.50 | 3 | 77.40 | 1.10 |
| 2 | 78.60 | 3 | 77.33 | 1.27 |
| 3 | 78.00 | 2 | 76.96 | 1.04 |
| | **78.37 ± 0.32** | | | **1.14 ± 0.12** |

Train loss tiếp tục giảm suốt 10 epoch. Đây là chữ ký overfitting kinh điển, và
nó xuất hiện rất sớm.

**[SUY]** Với 104 lớp huấn luyện và tập test toàn **lớp chưa thấy**, việc khớp
phân phối train làm hỏng chính cấu trúc ngữ nghĩa cần cho zero-shot. Đây là biểu
hiện của tension cố hữu: *thích nghi miền sketch* ↔ *giữ tính tổng quát của
CLIP*.

**[ĐO]** Hệ quả vận hành: mỗi run chỉ cần ~8 epoch thay vì 60 — **giảm 7.5× ngân
sách GPU**, và early stopping theo best epoch là bắt buộc chứ không phải tuỳ chọn.

### 3.2 Cross exchange: trung tính ở giá 272 lần

**[ĐO]** `78.41 → 78.24 = −0.17 pp`, nhỏ hơn δ_min = 0.31 pp.

**[ĐO]** Chi phí tham số ở `prompt_depth=12, cross_layer=6`:

| | params |
|---|---:|
| Tắt cross exchange | 169,984 |
| Bật cross exchange | 46,283,776 |
| | **272×** |

**[SUY]** Lý do cấu trúc nó không thể giúp: `visual_visual_learner.forward()`
**không nhận dữ liệu**. Nó là ánh xạ *tham số → tham số*, ảnh nằm trọn trong
`R^36864`. Tập giá trị sinh ra được không thể lớn hơn việc học thẳng 36,864 số.
Về capacity, đóng góp bằng **đúng 0**; lợi ích nếu có chỉ thuần về hình học tối
ưu hoá.

**[ĐO]** Hai khuyết tật khởi tạo, đo bằng mô phỏng lại đúng các phép init của
PyTorch:

- LKP (`AttentionPooling`) trả `ln(token_query + attn(...))`, mà `token_query`
  init `torch.randn` (std 1.0) còn prompt init std 0.02 — lệch 50×. Kết quả:
  proxy chứa **8.5% thông tin từ prompt**, 91.5% là chính token query ngẫu nhiên.
- `CrossPromptAttention` khuếch đại scale prompt **16×** (0.020 → 0.326), do hai
  nhánh residual đều có đầu vào bị LayerNorm chuẩn hoá về std 1.

**[SUY]** Khuyết tật thứ hai nhẹ hơn tôi từng đánh giá: prompt được tiêm vào
residual stream rồi mới qua `ln_1(x)`, nên LayerNorm chuẩn hoá từng token và
prompt vẫn tham gia attention đầy đủ bất kể độ lớn.

### 3.3 L4 không phải nguồn overfit

**[ĐO]** 3 seed, kiểm định cặp:

| | best | suy giảm |
|---|---:|---:|
| `lambda_ce = 0.0` | 77.54 ± 0.34 | 1.04 ± 0.63 |
| `lambda_ce = 1.0` | 78.37 ± 0.34 | 0.78 ± 0.36 |

Kiểm cặp: chênh **0.825 pp**, KTC 95% `[−0.934, −0.714]`, t = −14.79.

Bỏ L4 làm best **tệ hơn** và **không** giảm suy giảm. Giả thuyết bị bác bỏ.

**Cảnh báo về confound:** `lambda_ce=0` cũng làm prompt văn bản ngừng học hoàn
toàn (gradient của chúng chỉ đến từ L4 — CSV xác nhận `loss_ce = 0.0` chính
xác). Nên kết luận chặt là *"bỏ L4 có hại"*, chưa phải *"L4 không đóng vai trò
gì trong overfit"*.

### 3.4 Weight decay trơ về mặt số học

**[ĐO]** Cùng seed, chỉ đổi `weight_decay`:

| `wd` | best | lệch so với 0 |
|---|---:|---:|
| 0 | 0.785003 | — |
| 1e-4 | 0.785012 | +0.0009 pp |
| 1e-2 | 0.785141 | +0.0138 pp |

Lệch lớn nhất **nhỏ hơn δ_min 22 lần**.

**[GIẢ]** Cơ chế: `torch.optim.Adam` cộng L2 vào gradient **rồi mới** chuẩn hoá
theo `sqrt(v)` per-parameter, nên số hạng `wd·p` bị pha loãng. Với `lr = 1e-4` và
~3.600 step, lượng co lại không đáng kể. Kiểm chứng được bằng cách chuyển sang
**AdamW** (decoupled decay) — nhưng xem §3.5 trước khi đầu tư.

**Lưu ý phương pháp:** phép so ở Giai đoạn 2b vô tình đổi **hai** biến cùng lúc
(`lambda_ce` **và** `weight_decay`). May là bước sàng lọc 1 seed đã tách được, và
0.825 pp kia quy hoàn toàn về `lambda_ce`.

### 3.5 Capacity làm mọi thứ tệ hơn — kết quả mạnh nhất

**[ĐO]** 1 seed, 8 epoch, chỉ đổi `n_ctx`:

| `n_ctx` | params | epoch0 | best | last | **suy giảm** | best@ | lợi từ train |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 169,984 | 77.19 | 78.50 | 78.14 | **0.36** | 3 | +1.31 |
| 4 | 208,896 | 77.25 | 78.35 | 76.73 | **1.62** | 1 | +1.10 |
| 8 | 286,720 | 77.94 | 78.04 | 75.05 | **2.99** | 1 | +0.10 |
| 16 | 442,368 | 78.11 | 78.11 | 74.57 | **3.54** | 0 | **0.00** |

Ba xu hướng đơn điệu cùng lúc:

1. Suy giảm tăng **9.8 lần**
2. Best epoch lùi từ 3 về **0** — ở `n_ctx=16`, huấn luyện làm mô hình tệ đi
   **ngay từ epoch đầu tiên**
3. **Lợi ích từ huấn luyện tụt về 0** (best − epoch0: +1.31 → 0.00)

Và best mAP **không hề tăng**: 78.50 → 78.11.

**[SUY]** Capacity lớn hơn → khớp phân phối train nhanh hơn → chạm đỉnh sớm hơn
ở **cùng độ cao** → rồi rơi. Đây là bằng chứng trực tiếp nhất rằng bài toán là
tổng quát hoá chứ không phải sức biểu diễn.

Kết quả này **đảo ngược khuyến nghị ban đầu của tôi**. Tôi từng xếp "tăng
`n_ctx`" là ưu tiên số một với lập luận mô hình thiếu capacity vì prompt chỉ
chiếm 3.8% chuỗi ViT. Lập luận đó sai, và dữ liệu bác bỏ nó dứt khoát.

### 3.6 Trọng số trộn học được: α → 1, và điều đó làm tệ đi

**[ĐO]** Cho `α = sigmoid(θ)` học được (khởi tạo θ=0 → α=0.5, trùng đúng hành vi
cũ), 3 seed, 8 epoch:

| Cấu hình | α @ep1 | α @ep7 | best mAP | suy giảm |
|---|---:|---:|---:|---:|
| α cố định 0.5 | 0.500 | 0.500 | **78.37** | **0.74** |
| học, `lr=1e-3` | 0.545 | 0.780 | 77.86 | 1.38 |
| học, `lr=1e-2` | 0.718 | 0.926 | 77.57 | 1.43 |

<sub>α là trung bình của `α_photo` và `α_sketch` trên 3 seed.</sub>

Kiểm cặp: base tốt hơn **+0.507 pp** (KTC `[+0.414, +0.600]`, t=10.68) so với
`lr=1e-3`, và **+0.796 pp** (KTC `[+0.677, +0.917]`, t=12.85) so với `lr=1e-2`.

**Ba cột đơn điệu cùng chiều.** α tăng càng nhanh → best càng thấp, suy giảm càng
lớn. Đây là **quan hệ liều–đáp ứng**, bằng chứng nhân quả mạnh hơn hẳn tương quan
đơn thuần.

**[SUY]** Vì sao α → 1 là tất yếu, không phải phát hiện: nhánh frozen là hằng số
nên **không thể** giảm train loss; nhánh prompted thì có thể. Gradient descent
buộc phải dịch trọng số về phía thành phần thích nghi được.

> **Tổng quát:** bất kỳ trọng số học được nào giữa một thành phần **thích nghi**
> và một thành phần **đóng băng**, khi huấn luyện trên chính mục tiêu huấn luyện,
> đều sẽ dịch về phía thành phần thích nghi — bất kể điều đó tốt hay xấu cho tập
> test.

**[SUY]** Giá trị thật của thí nghiệm không phải "hỏi ý mô hình" (mô hình chỉ
phát biểu được về train loss, không về lớp chưa thấy) mà là: nó **đo được theo
thang liều lượng mức lệch pha giữa mục tiêu huấn luyện và mục tiêu đánh giá**,
chỉ bằng một tham số.

**[SUY]** Đảo ngược một kết luận trước đó: **residual mix cố định ở 0.5 là một
tính năng, không phải khiếm khuyết.** Nó là bộ điều chuẩn ngầm, và nó hoạt động
**chính vì không học được**.

**Bất đối xứng modality — bằng chứng yếu.** Tỉ lệ dịch chuyển của `α_sketch` so
với `α_photo` (đo từ mốc 0.5) ở epoch 1 là **1.30×** (`lr=1e-3`) và **1.35×**
(`lr=1e-2`), dương ở cả 3/3 seed — đúng chiều giả thuyết "CLIP yếu trên sketch".
Nhưng ở `lr=1e-3`, gap tan biến về cuối (epoch 7: −0.34 pp, dấu lẫn lộn giữa các
seed). Chưa đủ để làm một luận điểm.

### 3.6b Đường cong `mAP(α)`: hai nhánh BỔ TRỢ nhau, và 0.5 là đỉnh thật

**[ĐO]** Quét α tại thời điểm eval trên checkpoint tốt nhất (`α=0.5` tái lập
đúng 78.500 = con số checkpoint báo lúc train, nên bảng đáng tin):

| α | mAP@200 | P@200 | |
|---:|---:|---:|---|
| 0.00 | **30.014** | 26.322 | frozen-only — CLIP thuần |
| 0.10 | 44.781 | 40.942 | |
| 0.20 | 59.814 | 56.503 | |
| 0.30 | 71.348 | 68.103 | |
| 0.40 | 77.054 | 73.985 | |
| **0.50** | **78.500** | **75.690** | **đỉnh** — mặc định hiện tại |
| 0.60 | 78.315 | 75.624 | |
| 0.70 | 77.682 | 75.024 | |
| 0.85 | 76.701 | 73.959 | |
| 1.00 | **75.914** | 73.025 | prompted-only |

**Ba kết luận, tất cả đều bất ngờ so với dự đoán trước đó:**

**(1) Cơ chế prompt đóng góp +48.5 pp, không phải ~1.5 pp.** CLIP zero-shot thuần
chỉ đạt **30.0** mAP@200 — khoảng cách miền sketch↔photo của CLIP gốc lớn hơn
nhiều so với hình dung. Prompt tuning là thành phần chính của hệ thống, không phải
lớp tinh chỉnh mỏng.

> **⚠ Sửa một suy luận sai trong bản trước của tài liệu này.** Tôi từng viết:
> *"mAP sau một epoch đã là 77.19 còn đỉnh là 78.50 — toàn bộ huấn luyện chỉ thêm
> 1.31 pp"* và suy ra prompt có lẽ đóng góp rất ít. Sai lầm: `epoch0` là mAP **sau
> một epoch huấn luyện đầy đủ** (450 step), không phải trước khi huấn luyện. Prompt
> tuning hội tụ trong **chưa tới một epoch**, nên `epoch0` đã chứa gần như toàn bộ
> phần đóng góp. Manh mối đó chỉ sai hướng hoàn toàn.

**(2) `α = 0.5` là đỉnh thật, đã xác nhận bằng kiểm định.** Giá trị 1:1 trông như
tuỳ tiện lại nằm đúng cực đại. Kiểm cặp α=0.5 vs α=0.6: **+0.185 pp**, KTC 95%
`[+0.091, +0.278]`, t = 3.87 — **có ý nghĩa**. Lưu ý δ_min = 0.31 pp **không áp
dụng** ở đây: hai giá trị α dùng chung một checkpoint và chung đặc trưng, nên
không có dao động seed, chỉ còn bất định do lấy mẫu query.

**[ĐO] Một chi tiết quan trọng hơn cả kết luận.** Trong phép so đó:
`4,639 query tốt hơn / 4,977 kém hơn / 3,078 hoà`. **Nhiều query bị α=0.5 làm tệ
đi hơn là làm tốt lên**, vậy mà trung bình vẫn cao hơn — nghĩa là lợi thế đến từ
**thắng đậm trên một nhóm nhỏ**, không phải cải thiện đều tay.

**[SUY]** Nếu nhóm đó tập trung ở vài lớp cụ thể thì ta biết nhánh frozen đang bù
đắp **chính xác cái gì**, và đó là thông tin bắt buộc để thiết kế đúng regularizer.
Dùng `scripts/analyze_alpha_perclass.py` (chỉ cần numpy, đọc `ap_alpha*.npz` đã
lưu) — xem §8.

**(3) Hai nhánh BỔ TRỢ nhau, không chỉ là điều chuẩn.** Hỗn hợp vượt **cả hai**
thành phần: hơn prompted-only **+2.59 pp**, hơn frozen-only **+48.5 pp**. Nghĩa là
nhánh frozen mang thông tin mà nhánh prompted **đã đánh mất** trong quá trình thích
nghi — đúng là biểu hiện định lượng của tension ở §4.

**Đường cong bất đối xứng mạnh:** dốc `+14.5 pp` mỗi 0.1 α ở phía trái đỉnh, chỉ
`−1.9 pp` ở phía phải. Nên chệch lên trên 0.5 rẻ hơn nhiều so với chệch xuống dưới.

**[ĐO] Kiểm tra chéo với §3.6.** Dùng đường cong này dự đoán hiệu năng của các run
α học được, tại giá trị α ở epoch tốt nhất của chúng:

| Run | α tại best epoch | đường cong dự đoán | thực tế | lệch |
|---|---:|---:|---:|---:|
| `lr=1e-3`, s1 | 0.554 | 78.40 | 78.12 | −0.28 |
| `lr=1e-3`, s2 | 0.670 | 77.87 | 77.82 | −0.05 |
| `lr=1e-2`, s3 | 0.729 | 77.49 | 77.99 | +0.50 |

Hai thí nghiệm hoàn toàn độc lập khớp nhau trong ~0.3 pp. Điều này củng cố cả hai.

### 3.6c Phân tích theo lớp: α tối ưu **phụ thuộc nội dung**, và dự đoán được

**[ĐO]** Gain của hỗn hợp (α=0.5) so với prompted-only (α=1.0), theo từng lớp, kèm
SE của hiệu theo cặp trong lớp:

| Cần frozen nhất | gain | α* | | Bị hỗn hợp làm hại | gain | α* |
|---|---:|---:|---|---|---:|---:|
| scissors | +21.37 | 0.20 | | saw | −8.55 | 1.00 |
| window | +15.02 | 0.40 | | cabin | −8.23 | 1.00 |
| seagull | +10.12 | 0.30 | | songbird | −4.43 | 1.00 |
| bat | +9.15 | 0.40 | | skyscraper | −2.04 | 0.70 |

**19/21 lớp** có gain phân biệt được với 0 (SE điển hình 0.40 pp). **7 lớp bị hỗn
hợp làm TỆ đi.** α* biến thiên từ **0.20 đến 1.00**, std 0.263.

**[ĐO] Hỗn hợp là một đánh đổi, không phải cải thiện thuần:**

```
tổng phần được giúp  +3.85 pp
phần bị hại          −1.26 pp
ròng                 +2.59 pp
```

Con số +2.59 pp mà §3.6b báo là **phần còn lại** sau khi đã trả 1.26 pp, không
phải lợi ích đều tay.

**[ĐO] α* có cấu trúc dự đoán được:**

```
corr( chất lượng nhánh prompted , α*   ) = +0.42
corr( chất lượng nhánh prompted , gain ) = −0.55
```

| Nhóm | Số lớp | mAP của nhánh prompted |
|---|---:|---:|
| α* ≤ 0.4 — cần frozen bù | 8 | 70.1 |
| α* = 1.0 — chỉ dùng prompted | 5 | 86.3 |

> **Quy luật:** lớp mà nhánh prompted **đã tốt** thì muốn α cao (đừng pha loãng);
> lớp mà nhánh prompted **yếu** thì cần nhánh frozen bù.

**[SUY]** Đây là bằng chứng trực tiếp rằng **một hằng số α toàn cục là thoả hiệp**,
và rằng một **cổng α phụ thuộc nội dung** là khả thi về nguyên tắc — vì tín hiệu
quyết định (nhánh prompted có đáng tin cho đầu vào này không) có cấu trúc, không
phải ngẫu nhiên.

**[ĐO] Dư địa của α thích ứng, ba mức:**

| Mức | mAP@200 | so với α toàn cục | |
|---|---:|---:|---|
| α toàn cục tốt nhất (0.5) | 78.500 | — | đạt được |
| oracle theo **lớp** | 81.368 | **+2.868** | cần nhãn |
| oracle theo **từng query** | 83.818 | **+5.318** | trần tuyệt đối |

**[SUY]** Nhãn lớp giải thích **54%** dư địa; **46% còn lại nằm trong nội bộ từng
lớp**. Nghĩa là một cổng theo query, nếu tín hiệu đủ tốt, **có thể vượt** oracle
theo lớp — không bị nó chặn trên.

Con số này **lớn hơn mọi hiệu ứng đã đo được trong toàn bộ dự án** (mọi thí nghiệm
khác đều nằm trong ±0.8 pp). Đây là lần đầu có một dư địa đáng kể được xác định.

> **⚠ Sửa một lỗi khái niệm của tôi.** Tôi từng gọi oracle-theo-lớp là "cận trên
> của hướng H". Sai: nó **không chặn trên** một cổng theo *query*, vì α* còn biến
> thiên trong nội bộ từng lớp. Trần tuyệt đối là **oracle theo từng query**. Script
> nay báo cả ba mức: α toàn cục (đạt được) / oracle theo lớp (cần nhãn) / oracle
> theo query (trần tuyệt đối).

**Câu hỏi quyết định còn lại:** dư địa đó có **với tới được bằng tín hiệu quan sát
được lúc suy luận** không? `sweep_alpha.py` nay lưu thêm các tín hiệu không cần
nhãn — độ đồng thuận hai nhánh `u·f`, độ nhọn top-1 của mỗi nhánh, và hiệu của
chúng. `analyze_alpha_perclass.py` chia query theo phân vị từng tín hiệu rồi chọn
α oracle cho mỗi nhóm: đó là **trần của mọi cổng học từ tín hiệu ấy**.

**Chạy lại cả hai script để có con số này** — nếu tín hiệu tốt nhất chỉ với tới
vài phần trăm của trần thì hướng H không khả thi dù dư địa lớn.

### 3.7 Không có tập validation riêng — best epoch chọn trên tập test

**[ĐO]** `ValidDataset` (`src/dataset_retrieval.py`) dùng đúng `UNSEEN_CLASSES`,
cùng tập lớp dùng để báo cáo kết quả ZS-SBIR. `ModelCheckpoint` monitor
`val_map_200`, và `best_metric` là `max` theo epoch trên chính tập đó.

**[SUY]** Với suy giảm best−last = 1.14 pp và best epoch = 2–3 trong 8–10 epoch,
việc chọn epoch **có nhìn tập test** mang lại thiên lệch lạc quan cỡ vài phần
mười pp, chặn trên bởi 1.14 pp.

**Phạm vi ảnh hưởng:** mọi cấu hình trong tài liệu này đều chịu **cùng một** thiên
lệch, nên các **so sánh** (toàn bộ §3.2–3.6) vẫn hợp lệ. Chỉ **con số tuyệt đối**
là lạc quan — quan trọng khi đưa vào bảng so sánh với các công trình khác.

---

## 4. Chẩn đoán tổng hợp

Bốn mũi can thiệp, bốn kết quả:

| Can thiệp | Chiều | Kết quả |
|---|---|---|
| **Bỏ** một số hạng loss (L4) | ↓ thích nghi | tệ hơn 0.83 pp |
| **Thêm** regularization tổng quát | ↓ thích nghi | trơ (0.014 pp) |
| **Thêm** capacity (`n_ctx`) | ↑ thích nghi | tệ hơn nhiều (suy giảm ×9.8) |
| **Nới** ràng buộc residual mix (α học được) | ↑ thích nghi | tệ hơn 0.51–0.80 pp |

**[SUY]** Hai can thiệp làm **tăng** mức thích nghi đều làm tệ đi, đơn điệu theo
cường độ. Nghĩa là điểm vận hành hiện tại đã nằm ở **phía thích nghi quá mức**
của đường cong. Đây là chẩn đoán chính, và nó chặt hơn kết luận trước đó nhờ có
thêm §3.6.

**[SUY]** Suy giảm **không đến từ một khuyết tật cụ thể nào** mà là hệ quả nội
tại của việc khớp phân phối huấn luyện — trên 104 lớp **đã thấy**, trong khi test
là lớp **chưa thấy**.

**[ĐO+SUY]** Cơ chế giữ tính tổng quát là **residual mix**:

```python
feat = norm( norm(prompted) + frozen )
```

§3.6 cho thấy nó hiệu quả **chính vì** tỉ lệ 1:1 là hằng số áp đặt. Cho mô hình
tự chọn thì nó chọn sai, một cách tất yếu và có thể dự đoán được.

**[ĐO] §3.6b làm rõ bản chất của nó: đây là BỔ TRỢ, không chỉ là điều chuẩn.**
Hỗn hợp (78.50) vượt **cả hai** thành phần — prompted-only (75.91) và frozen-only
(30.01). Nhánh frozen mang thông tin mà nhánh prompted đã đánh mất khi thích nghi.
Và α = 0.5 nằm đúng cực đại của đường cong.

**[SUY]** Do đó lời giải **không** phải là nới ràng buộc này ra (§3.6 và §3.6b
bác bỏ) và cũng **không** phải chỉnh lại hằng số α (§3.6b cho thấy 0.5 đã tối ưu).
Chỉ còn một hướng:

> Làm cho **bản thân quá trình thích nghi bảo toàn tính tổng quát hơn**, để nhánh
> prompted không đánh mất thứ mà nhánh frozen đang phải bù lại.

**[SUY]** Điều này còn có một hệ quả thực tiễn đo được: hiện mỗi ảnh phải qua
**hai** lượt forward ViT (prompted + frozen) khi suy luận. Nếu nhánh prompted tự
đạt được 78.5 thì **chi phí suy luận giảm một nửa**. Hiện nó chỉ đạt 75.91, tức
khoảng cách cần lấp là **2.59 pp**.

---

## 5. Hướng phát triển

Mỗi hướng ghi rõ **cơ sở** (bằng chứng nào dẫn tới nó), **cơ chế**, **chi phí**,
và **cách bác bỏ** — nếu một đề xuất không nêu được cách nó có thể sai thì nó
chưa phải giả thuyết khoa học.

### A. ~~Quét α cố định~~ — **ĐÃ XONG, không có lợi ích miễn phí** ✔

**Cơ sở.** §3.6 cho biết α=0.5 tốt hơn α→1, đơn điệu. Nhưng **chưa có dữ liệu nào
về α < 0.5**. Đường cong `mAP(α)` là thứ rẻ nhất còn lại và trả lời nhiều câu hỏi
treo nhất cùng lúc.

**Cơ chế.** Cố định α, không học. Quét `α ∈ {0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}`.

**Ba câu hỏi được trả lời cùng lúc:**

| Điểm trên đường cong | Ý nghĩa |
|---|---|
| **α = 0** | chính là `frozen-only` — con số còn thiếu từ Giai đoạn 0 |
| **vị trí đỉnh** | nếu đỉnh ở α<0.5, nhánh prompted nên bị hạ trọng số |
| **độ dốc quanh 0.5** | kết quả nhạy thế nào với lựa chọn này |

**Chi phí.** Quét tại **thời điểm eval** trên một checkpoint đã train: 1 run
(~40 phút) + N lần eval (vài phút mỗi lần). Quét tại thời điểm train thì đắt hơn
(N run) nhưng cho biết tối ưu chung của (train, eval).

**Kết quả (§3.6b).** Đỉnh nằm đúng tại **α = 0.5** — giá trị mặc định. Không có
lợi ích miễn phí nào. Nhưng đường cong trả về ba thông tin quan trọng hơn cả câu
hỏi ban đầu:

- frozen-only = **30.0** → cơ chế prompt đóng góp **+48.5 pp**, không phải ~1.5 pp
- hỗn hợp vượt cả hai thành phần → **bổ trợ**, không chỉ điều chuẩn
- prompted-only = **75.9**, tức khoảng cách 2.59 pp cần lấp nếu muốn bỏ nhánh
  frozen khỏi suy luận

Theo cây quyết định đã đặt ra: *"đỉnh ở α ≈ 0.5 → ưu tiên hướng B′"*.

### B. ~~Thay residual mix bằng ràng buộc quan hệ~~ — **CHỐNG CHỈ ĐỊNH**

**Trạng thái trước §3.6:** tôi xếp hướng này "trần cao nhất".

**Vì sao rút lại.** Bỏ residual mix để nhánh prompted làm tất cả **chính là** cực
hạn α → 1. §3.6 đo được rằng đi theo hướng đó, dù chỉ một phần, làm hiệu năng tệ
đi **đơn điệu theo cường độ**: α@ep7 = 0.78 → −0.51 pp; α@ep7 = 0.93 → −0.80 pp
(mAP thấp hơn base).
Ngoại suy tới α = 1.0 không có lý do gì để tốt hơn.

`L_rel` **có thể** làm thay đổi bức tranh này — nó là ràng buộc mà thí nghiệm α
không có. Nhưng giờ nó phải vượt qua một bằng chứng phản bác trực tiếp thay vì
chỉ là giả thuyết trung tính. **Không nên đầu tư trước khi có đường cong ở
hướng A.**

**Điều kiện hồi sinh:** nếu §A cho thấy đỉnh nằm ở α > 0.5, thì giả định nền của
hướng B được khôi phục và đáng thử lại.

### B′. `L_rel` như ràng buộc BỔ SUNG, giữ nguyên α = 0.5 — *ưu tiên số một hiện nay* ⭐

**Cơ sở.** §4: hướng duy nhất còn lại sau khi §3.6 bác bỏ việc nới ràng buộc và
§3.6b cho thấy hằng số α đã tối ưu. §3.6 bác bỏ việc **nới** ràng buộc, nhưng
không nói gì về việc **thêm** ràng buộc.

**Mục tiêu định lượng rõ ràng từ §3.6b.** Không cần hứa hẹn mơ hồ: nhánh prompted
hiện đạt **75.91**, hỗn hợp đạt **78.50**. Khoảng cách **2.59 pp** chính là lượng
thông tin mà nhánh prompted đánh mất khi thích nghi. `L_rel` thành công nghĩa là
thu hẹp khoảng cách đó.

> **⚠ Rút lại một tuyên bố quá lạc quan trong bản trước.** Tôi từng viết rằng
> "mức thành công mạnh" của B′ là đưa `prompted-only` lên ≈78.5 để **bỏ hẳn nhánh
> frozen, giảm một nửa chi phí suy luận**. Phân tích kỹ cho thấy **không đường nào
> trong hai đường hiển nhiên làm được điều đó**:
>
> - **Ép `sim(F_prompted) = sim(F_frozen)`:** với vector đã chuẩn hoá, khớp *mọi*
>   tích vô hướng đôi một buộc `F_prompted` phải là một **phép quay trực giao** của
>   `F_frozen`. Retrieval chỉ dùng tích vô hướng, nên hiệu năng bằng đúng
>   frozen-only = **30.0**. `L_rel` mạnh kéo prompted **xuống**, không lên.
> - **Tự chưng cất hỗn hợp vào nhánh prompted** (`p ← norm(0.5p + 0.5f)`): điểm bất
>   động là `p ∥ f`, tức cũng sụp về frozen.
>
> **Lý do gốc:** hỗn hợp tốt **chính vì hai nhánh khác nhau**. Mọi mục tiêu ép
> chúng giống nhau đều phá bỏ đúng cái nguồn lợi ích đó.

**Mục tiêu thực tế của B′, đã điều chỉnh:** `L_rel` là một **núm đánh đổi**, không
phải cải thiện miễn phí. λ=0 cho thích nghi tự do (prompted-only = 75.91);
λ→∞ cho tương đương frozen (30.0). Câu hỏi là liệu có tồn tại λ ở giữa mà tại đó
**hỗn hợp vượt 78.50** — tức nhánh prompted giữ lại nhiều cấu trúc hơn mà vẫn thích
nghi đủ.

| Kết quả | Ý nghĩa |
|---|---|
| hỗn hợp > 78.50 | `L_rel` là regularizer đúng hướng — kết quả dương đầu tiên của cả dự án |
| hỗn hợp ≈ 78.50, prompted-only tăng | đánh đổi tốc độ/độ chính xác đo được, không miễn phí |
| hỗn hợp giảm đơn điệu theo λ | giả thuyết "cấu trúc quan hệ là thứ cần giữ" sai |

**Điều kiện tiên quyết:** chạy `analyze_alpha_perclass.py` trước. Nếu lợi ích của
nhánh frozen tập trung ở vài lớp cụ thể thì một `L_rel` áp đều toàn cục là sai
thiết kế ngay từ đầu.

**Cơ chế.** Giữ nguyên residual mix cố định, cộng thêm:

```
L_rel = ‖ sim(F_prompted) − sim(F_frozen) ‖²_F
```

trên ma trận tương đồng trong batch. Ép nhánh prompted giữ hình học tương đối của
CLIP **trong khi vẫn được ghìm bởi residual mix**.

**Dự đoán kiểm chứng được [GIẢ].** Nếu `L_rel` thật sự bảo toàn tính tổng quát
thì **suy giảm phải giảm** — đây là chỉ số nhạy nhất ta có (std 0.12 pp). Và khi
đó, α học được sẽ tăng chậm hơn: một phép kiểm chéo độc lập.

**Cách bác bỏ.** Nếu suy giảm không đổi với mọi trọng số `L_rel`, giả thuyết
"cấu trúc quan hệ là thứ cần giữ" sai.

### G. Tách lớp `pseudo-unseen` để chọn mô hình — *mới, sinh ra từ §3.6 + §3.7* ⭐

**Cơ sở.** Hai kết quả độc lập cùng chỉ về một chỗ:

- §3.6: mục tiêu huấn luyện **không thể** cho biết điều gì tốt cho lớp chưa thấy
  — nó đẩy α đi sai hướng một cách tất yếu.
- §3.7: hiện **không có tập validation riêng**; best epoch chọn trên chính tập
  test.

Cả hai đều là biểu hiện của **cùng một thiếu sót**: không có tín hiệu nào đại
diện cho "lớp chưa thấy" mà thuật toán được phép nhìn.

**Cơ chế.** Chia 104 lớp huấn luyện thành `seen-train` (~84) và `pseudo-unseen`
(~20). Huấn luyện trên `seen-train`; dùng `pseudo-unseen` để chọn epoch, chọn α,
và chọn mọi siêu tham số.

**Vì sao đây là hướng mạnh:**

1. **Sửa lỗi phương pháp.** Loại bỏ việc chọn mô hình trên tập test (§3.7) — cần
   thiết cho bất kỳ báo cáo nghiêm túc nào.
2. **Mở khoá lựa chọn có nguyên tắc.** α, số epoch, cường độ thích nghi — tất cả
   đều là những đại lượng mà §3.6 chứng minh không thể chọn từ train loss. Giờ
   chúng có một mục tiêu đại diện hợp lệ.
3. **Có thể nâng thành đóng góp.** Không chỉ *chọn* α trên `pseudo-unseen` mà
   **học** nó ở đó — một mục tiêu bi-level nhắm thẳng vào sự lệch pha đã đo được.
   §3.6 đã cho thấy học α trên tập train là sai; học trên pseudo-unseen là phiên
   bản đúng của cùng ý tưởng.

**Chi phí.** Sửa `dataset_retrieval.py` để tách lớp, thêm một val loader. Vừa
phải. Tập train nhỏ đi ~20% — cần đo lại baseline.

**Cách bác bỏ.** Nếu α chọn trên `pseudo-unseen` ≈ α chọn trên test, thì
`pseudo-unseen` không mang thêm thông tin và chỉ còn giá trị về tính liêm chính
của phương pháp (vẫn đáng làm, nhưng không phải đóng góp).

### H. Cổng α phụ thuộc nội dung, huấn luyện trên `pseudo-unseen` — *mới, từ §3.6c* ⭐

**Cơ sở — bốn phép đo độc lập cùng chỉ về đây:**

| # | Bằng chứng | Nói lên điều gì |
|---|---|---|
| §3.6c | α* biến thiên 0.20–1.00 theo lớp (std 0.263) | hằng số toàn cục là thoả hiệp |
| §3.6c | `corr(chất lượng prompted, α*) = +0.42` | tín hiệu quyết định **có cấu trúc**, dự đoán được |
| §3.6c | hỗn hợp trả 1.26 pp để lấy 3.85 pp | cổng hoàn hảo giữ phần lợi, tránh phần hại |
| §3.6 | α học trên mục tiêu train luôn → 1 | **không** được huấn luyện cổng trên tập train |

**Cơ chế.**

```
α(x) = σ( g([u ; f ; u·f]) ),   u = prompted_norm,  f = frozen_norm
feat  = norm( α(x)·u + (1−α(x))·f )
```

`g` là MLP nhẹ (1025 → 64 → 1), vài chục nghìn tham số. Tín hiệu vào **có sẵn lúc
suy luận và không cần nhãn** — đó là điểm mấu chốt phân biệt với oracle theo lớp.
Độ bất đồng `u·f` là ứng viên tự nhiên: khi hai nhánh đồng thuận, trộn thế nào cũng
vậy; khi chúng lệch nhau, mới cần quyết định tin ai.

**Phụ thuộc bắt buộc vào hướng G.** §3.6 đã chứng minh học α trên mục tiêu huấn
luyện thì nó chạy về 1 — một cách tất yếu, vì nhánh frozen không giảm được train
loss. Cổng `g` sẽ mắc **đúng** lỗi đó. Nên nó **phải** được huấn luyện trên tập
lớp `pseudo-unseen` (hướng G). Điều này biến G từ "sửa lỗi phương pháp" thành
**thành phần cấu tạo** của phương pháp.

**Cận trên.** Oracle α-theo-lớp (§3.6c) chặn trên mọi cách dự đoán α. **Phải đo
trước** — nếu oracle chỉ hơn α toàn cục vài phần mười pp thì bỏ hướng này ngay.

**Cách bác bỏ.** (a) oracle ≈ α toàn cục; hoặc (b) cổng huấn luyện trên
`pseudo-unseen` không tổng quát sang lớp test thật.

**Vì sao đây là đóng góp mạnh hơn B′.** Nó xuất phát từ một quan sát đo được
(§3.6c) chứ không từ suy đoán, nó giải thích được **vì sao** phải có G, và nó
không mắc bẫy "ép hai nhánh giống nhau" đã làm hỏng B′.

### C. Prompt điều kiện theo instance — *cứu cross exchange*

**Cơ sở.** §3.2: ánh xạ tĩnh không phụ thuộc dữ liệu nên **không thể** thêm
capacity — đã chứng minh cả bằng lập luận lẫn 272× tham số cho 0 lợi ích.

**Cơ chế.** Biến ánh xạ từ *params→params* thành *input→prompts*:

```
prompt_i = base_i + g_i( f_frozen(input) )
```

`f_frozen` **đã được tính sẵn** mỗi step cho residual mix. `g_i` là MLP nhẹ
512→768, tổng ~4.7M tham số — **1/10 mapper hiện tại**.

**Giá trị narrative.** Kết quả âm của bạn trở thành motivation có bằng chứng:
*"ánh xạ prompt tĩnh không thêm capacity (chứng minh + ablation 46.3M tham số
trung tính), thay bằng ánh xạ điều kiện instance rẻ hơn 10 lần"*.

**Rủi ro [ĐO].** §3.5 và §3.6 đều cho thấy tăng mức thích nghi làm tệ đi. Prompt
điều kiện instance là **thêm capacity và thêm khả năng thích nghi** — tức đúng
chiều đã bị bác bỏ hai lần. Sau các kết quả này, ưu tiên của hướng C **giảm
mạnh**: chỉ nên thử **sau khi** B′ hoặc G cho thấy đã kiểm soát được tổng quát
hoá, và không bao giờ chạy độc lập.

### D. Sửa công thức text enhance — *chi phí gần 0*

**Cơ sở.** Phân tích cấu trúc loss hiện tại, chưa có [ĐO] vì `--enhance_text`
chưa từng bật trong các run đã chạy.

**Hai vấn đề.**

1. `1 − cos(a+b, a)` với `a`, `b` chuẩn hoá bằng `1 − √((1+a·b)/2)` — chỉ là
   reparameterization đơn điệu của cosine với gradient bão hoà. Không sai, nhưng
   không giải thích được trong bài báo. Viết thẳng `1 − cos(a,b)`.

2. `loss_cons_visual_cross` kéo đặc trưng **ảnh** về đặc trưng **text**. CLIP có
   **modality gap** đã được ghi nhận rộng rãi: hai loại đặc trưng nằm trong hai
   nón tách rời, `cos(image, text) ≈ 0.2–0.3` ngay cả với cặp khớp hoàn hảo. Ép
   căn chỉnh tuyệt đối qua khe hở đó đẩy đặc trưng ảnh ra khỏi manifold của CLIP.

**Cách sửa.** Dùng mục tiêu **tương đối**, chỉ so sánh **giữa các lớp**:

```
L = KL( softmax(f_sk · Tᵀ / τ) ‖ softmax(f_sk^frozen · Tᵀ / τ) )
```

`T` là ma trận prototype text GPT. Tôn trọng modality gap, vẫn truyền được tri
thức ngữ nghĩa từ mô tả LLM. **Cùng họ với hướng B** — cả hai đều là ràng buộc
quan hệ thay cho ràng buộc tuyệt đối.

**Nâng cấp.** Mô tả GPT riêng cho từng modality (`"a sketch of X: black-and-white
line drawing, no texture..."` vs `"a photo of X: ..."`) và dùng **hiệu vector**
giữa hai prototype như hướng dịch chuyển miền tường minh. Đây là tín hiệu duy
nhất trong toàn hệ thống mô tả *sketch khác photo ở đâu*, hiện đang bị bỏ phí.

### E. Sửa false negative trong loss contrastive — *chi phí ~0*

**Cơ sở.** Phân tích `cross_loss`: NT-Xent chuẩn, mỗi sketch chỉ **một** positive
là ảnh ghép cặp; mọi ảnh khác trong batch là negative, **kể cả ảnh cùng lớp**.

**[SUY]** Batch 128 / 104 lớp: mỗi hàng có 254 ứng viên, ~2.4 cùng lớp bị coi là
negative — chỉ ~1% về số lượng, **nhưng là những negative giống nhất** nên chi
phối gradient. Đây chính là thứ supervised contrastive sinh ra để sửa.

**Rủi ro.** Novelty thấp. Nên coi là cải tiến kỹ thuật đi kèm, không phải
contribution.

### F. Re-ranking phía retrieval — *không cần train lại*

**Cơ sở.** Thực hành chuẩn trong person-ReID, ít dùng trong SBIR.

k-reciprocal re-ranking / query expansion trên ma trận tương đồng cuối. Vài phút.
Phải báo cáo thành **dòng riêng** và kiểm xem benchmark có cho phép truy cập
toàn bộ gallery hay không.

---

## 6. Điều KHÔNG nên làm — và cơ sở

Phần này giá trị ngang phần đề xuất, vì mỗi mục đều có bằng chứng bác bỏ.

| Đừng làm | Cơ sở |
|---|---|
| Tăng `n_ctx` để có thêm capacity | §3.5 — suy giảm tăng 9.8×, lợi từ train về 0, best không tăng |
| **Cho α học được trên mục tiêu huấn luyện** | §3.6 — α→1 tất yếu, mAP tệ hơn 0.51–0.80 pp, liều–đáp ứng đơn điệu |
| **Bỏ residual mix để nhánh prompted làm tất cả** | §3.6 — đó là cực hạn α→1, chiều đã bị bác bỏ |
| Thêm weight decay để chống overfit | §3.4 — lệch 0.014 pp, dưới δ_min 22 lần |
| Bỏ L4 để giảm overfit | §3.3 — best tệ hơn 0.83 pp, KTC `[−0.93, −0.71]` |
| Giữ static cross exchange làm base | §3.2 — trung tính ở giá 272× tham số, và làm chậm mọi vòng lặp |
| Chạy 60 epoch | §3.1 — best epoch 2–3; 8 epoch là đủ, tiết kiệm 7.5× |
| **Báo cáo mAP tuyệt đối mà không nói rõ cách chọn epoch** | §3.7 — best epoch chọn trên tập test |
| Kết luận từ chênh lệch mAP < 0.31 pp | §2.1 — đó là δ_min |
| So hai cấu hình bằng 1 seed | §2.1 — std seed 0.32 pp |
| Đọc cột "n seed" như số seed khi có run chạy lại | Huấn luyện tất định → bản trùng làm std nhỏ đi giả tạo |

**Nguyên tắc rút ra từ §3.5 + §3.6:** mọi đề xuất làm **tăng** khả năng thích nghi
(thêm capacity, nới ràng buộc, thêm điều kiện hoá) đều đang đi ngược hai kết quả
liều–đáp ứng độc lập. Nếu đề xuất một hướng như vậy, phải nêu được **cơ chế bù**
giữ tổng quát hoá đi kèm.

---

## 7. Điều còn thiếu

### 7.1 mAP frozen-only — số liệu quan trọng nhất còn thiếu

Chạy `--eval_frozen_only` cho biết CLIP đóng băng thuần đạt bao nhiêu. Nó quyết
định toàn bộ thứ tự ưu tiên:

| Kết quả | Diễn giải | Hệ quả |
|---|---|---|
| ~77–78 | prompt mua ~0.5–1.5 điểm | Câu chuyện đổi thành *"cơ chế prompt hiện tại gần như không đóng góp"*; hướng B là hướng duy nhất đáng theo |
| ~70–73 | prompt mua ~5–8 điểm | Cơ chế prompt có giá trị thật; A và C đáng đầu tư đầy đủ |

**[ĐO]** Manh mối gián tiếp: ở `n_ctx=2`, mAP sau **một** epoch đã là 77.19 còn
đỉnh là 78.50 — toàn bộ huấn luyện chỉ thêm **1.31 pp** so với sau một epoch.
Con số này khiến nhánh thứ nhất của bảng trên có vẻ khả dĩ hơn.

### 7.2 L1 (InfoNCE) có phải nguồn overfit còn lại?

§3.3 loại L4, nhưng cấu hình `lambda_ce=0` (chỉ còn L1) **vẫn suy giảm 1.04 pp**.
Chưa có thí nghiệm nào cô lập L1.

### 7.3 Đường cong `mAP(α)` phía α < 0.5

§3.6 chỉ đo được α ≥ 0.5 (vì α học được luôn tăng). Nửa còn lại của đường cong
hoàn toàn chưa biết, và `α = 0` chính là §7.1. Đây là lý do hướng A gộp cả hai.

**[ĐO] Mốc tự kiểm tra cho `scripts/sweep_alpha.py`.** Cấu hình base (seed 1) đã
tái lập `best mAP = 0.785003 @epoch 3` ở **năm** lần chạy độc lập, trải qua các
giai đoạn 0, 2, 3, 4 dưới các `exp_name` khác nhau. Do đó:

> Khi quét α, giá trị tại **α = 0.5 phải bằng 78.5003**. Lệch đáng kể nghĩa là
> nạp sai checkpoint hoặc khác dữ liệu — **không** phải phát hiện khoa học.

**[SUY]** Năm lần tái lập chính xác cũng củng cố §2.1: huấn luyện tất định hoàn
toàn theo seed, nên **δ_min = 0.31 pp là dao động seed thuần tuý**, không lẫn
nhiễu chạy lại. Hệ quả thực tế: không bao giờ cần lặp lại một run cùng seed.

---

## 8. Thứ tự đề xuất

| # | Việc | Chi phí | Trả lời được gì |
|---|---|---|---|
| ✔ | ~~Hướng A~~ — quét α | xong | §7.1 + §7.3; α*=0.5, không có lợi ích miễn phí |
| ✔ | ~~`analyze_alpha_perclass.py`~~ | xong | §3.6c: α* phụ thuộc nội dung, dự đoán được |
| **0** | **Oracle α-theo-lớp** | **~1 phút, không GPU** | Cận trên của hướng H — quyết định có theo hay bỏ |
| **1** | **Hướng G + H** — `pseudo-unseen` + cổng α | ~2 ngày | Hiện thực hoá dư địa mà oracle chỉ ra |
| 2 | **Hướng B′** — `L_rel` bổ sung | ~4 giờ | Có λ nào đưa hỗn hợp vượt 78.50 không |
| 3 | **D, E, F** | rẻ | Cải tiến đi kèm, không phụ thuộc nhánh |
| — | ~~B~~, C | — | Hoãn: chống chỉ định / rủi ro cao sau §3.6 |

**Vì sao H vượt lên trước B′.** B′ sinh ra từ suy luận cấu trúc và đã phải rút lại
một nửa (xem cảnh báo trong mục B′). H sinh ra từ bốn phép đo độc lập trong §3.6c,
và nó giải thích được vì sao G là bắt buộc chứ không chỉ là dọn dẹp phương pháp.

**Chạy song song được:** hướng F (re-ranking) dùng đúng ma trận tương đồng mà
`sweep_alpha.py` đã tính, không cần train lại và không phụ thuộc kết quả nào khác.

**Mốc để đo hướng B′** — cả ba từ §3.6b, trên cùng một checkpoint:

```
frozen-only    30.01     nhánh đóng băng một mình
prompted-only  75.91     nhánh thích nghi một mình   ← cần nâng lên
hỗn hợp α=0.5  78.50     hiện tại                    ← mục tiêu
```

Nếu `L_rel` đưa `prompted-only` lên ~78.5 thì bỏ được nhánh frozen khỏi suy luận:
**cùng độ chính xác, một nửa compute**. Đó là phát biểu đóng góp trung thực nhất
mà dữ liệu hiện có cho phép.

---

## 8b. KẾT QUẢ DƯƠNG ĐẦU TIÊN: hậu xử lý retrieval (+3.256 pp)

**[ĐO]** Trên cùng checkpoint, không train lại gì:

| Phương pháp | mAP@200 | Δ | Cần nhãn lúc test? |
|---|---:|---:|---|
| Nền (α=0.5) | 78.500 | — | — |
| **+ αQE (`qe_k=20`)** | **81.756** | **+3.256** | **Không** |
| + α theo cụm (`k=42`) | 79.315* | +0.970* | Có |
| + cả hai | 82.071* | +3.725* | Có |

<sub>* chấm trên nửa query giữ lại, so với nền cũng trên nửa đó (78.346).</sub>

**Đây là cải thiện lớn nhất toàn dự án** — gấp 10 lần δ_min = 0.31 pp, và lớn hơn
mọi hiệu ứng ở §3.2–3.6 cộng lại.

**[ĐO] αQE đạt đỉnh tại `qe_k=20`** (5→20 tăng, 40→120 giảm dần), nên đây là cực
đại thật chứ không phải biên dải quét. **DBA luôn có hại** — `dba_k>0` với `qe_k=0`
cho âm ở mọi mức, và mọi tổ hợp có DBA đều kém hơn αQE thuần.

**[SUY] Vì sao hướng này thành công trong khi bảy hướng trước thất bại:** nó
**không đụng gì tới quá trình thích nghi**. Mọi can thiệp trước đều dịch chuyển
điểm cân bằng thích nghi ↔ tổng quát, mà §4 cho thấy điểm đó đã tối ưu. αQE tác
động lên một trục hoàn toàn khác — dùng cấu trúc của gallery, thứ chưa ai khai
thác.

### Phân biệt bắt buộc khi báo cáo

| | Dùng gì lúc test | Xếp loại |
|---|---|---|
| αQE | chỉ gallery (dữ liệu bài toán đã cho) | **transductive, không nhãn** — triển khai được |
| α theo cụm | gallery **+ nhãn** để chọn α mỗi cụm | chưa triển khai được |

Bảng trong bài báo phải tách dòng:

```
Ours (inductive)             78.5
Ours + αQE (transductive)    81.8
```

Phần lớn công trình SBIR không dùng hậu xử lý, nên gộp chung là so sánh không
công bằng.

### α theo cụm: dư địa thật, nhưng chưa dùng được

Cột giữ lại (79.315) rất sát cột oracle (79.597) → **cấu trúc cụm tổng quát hoá
thật**, không phải overfit. Đỉnh ở `k=42`, tức **gấp đôi số lớp thật (21)** — khớp
với §3.6c rằng 46% dư địa nằm trong nội bộ lớp.

**Còn thiếu để triển khai:** cách chọn α cho mỗi cụm **không dùng nhãn**. §3.6c cho
thấy tín hiệu theo từng query chỉ với tới 3%, nhưng trung bình theo cụm gộp ~300
query nên nhiễu giảm ~17 lần — tín hiệu có thể lộ ra ở mức cụm. **Chưa kiểm.**

---

## 9. Kế hoạch: từ bảy kết quả âm đến một bài báo

### 9.1 Đọc đúng tình hình

Bảy can thiệp, không cái nào cho lợi ích dương có ý nghĩa:

| # | Can thiệp | Kết quả |
|---|---|---|
| 1 | Cross exchange (46.1M params) | trung tính |
| 2 | Bỏ L4 | −0.83 pp |
| 3 | Weight decay | trơ (0.014 pp) |
| 4 | Tăng `n_ctx` | suy giảm ×9.8 |
| 5 | α học được | −0.51 đến −0.80 pp |
| 6 | Quét α cố định | 0.5 đã tối ưu |
| 7 | Cổng α theo tín hiệu query | 3% dư địa |

**[SUY]** Đây không phải chuỗi xui. Bảy phép đo độc lập vẽ ra **cùng một bức
tranh**: điểm vận hành hiện tại đã nằm ở phía *thích nghi quá mức*, và không có
đòn bẩy dễ nào còn lại. Giá trị đã tạo ra nằm ở **những gì đã loại trừ được**, và
đó là thứ đăng báo được.

### 9.2 Hai khung bài báo

**Khung A — bài phân tích/đo lường** (khả thi ngay với dữ liệu đã có):

> *"Điều gì thực sự quyết định hiệu năng trong prompt tuning dựa trên CLIP cho
> ZS-SBIR"*

Đóng góp, tất cả đã đo xong:

1. **Prompt tuning đóng góp +48.5 pp** (30.0 → 78.5) — thiết lập lại mốc nền, vì
   phần lớn công trình không báo cáo zero-shot CLIP thuần.
2. **Ánh xạ prompt tĩnh là reparameterization**, capacity thêm vào bằng 0 — chứng
   minh giải tích **và** ablation 46.1M tham số trung tính.
3. **Residual mix là bổ trợ, không phải điều chuẩn** — hỗn hợp vượt cả hai nhánh;
   α=0.5 là cực đại đã kiểm định.
4. **Trọng số hoà trộn học được luôn hỏng** — chứng minh cơ học + quan hệ liều–đáp
   ứng đo được. Áp dụng cho mọi kiến trúc có nhánh thích nghi + nhánh đóng băng.
5. **Capacity làm tệ đi đơn điệu** — `n_ctx` 2→16.
6. **Phương pháp luận:** giao thức δ_min, kiểm định cặp trên AP từng query, chỉ số
   suy giảm, và cảnh báo oracle-theo-query là thiên lệch chọn-trên-nhiễu.

Điểm (4) và (6) có giá trị vượt ra ngoài SBIR.

**Khung B — cần ít nhất một kết quả dương.** Ứng viên còn lại: H′, F, G, D, E.

### 9.3 H′ — α theo CỤM, phục hồi từ thất bại của H ⭐

**Cơ sở.** Cổng theo query thất bại (3%) **nhưng** cấu trúc mức lớp là thật
(+2.868 pp, sống sót kiểm chứng chia đôi). Vấn đề không phải "không có cấu trúc"
mà là **không đọc được cấu trúc lớp từ đặc trưng của một query riêng lẻ**.

**[SUY]** ZS-SBIR cho ta **toàn bộ gallery** lúc test. Có thể phục hồi cấu trúc lớp
bằng **phân cụm không giám sát** trên gallery, rồi gán α theo cụm. Đây là thiết lập
transductive — hợp lệ và phổ biến trong retrieval, chỉ cần báo cáo rõ.

**Kiểm được bằng dữ liệu đã lưu, không cần train:** k-means trên đặc trưng frozen
của gallery → gán mỗi query vào cụm gần nhất → oracle α theo cụm **kèm kiểm chứng
chia đôi**. Nếu phục hồi được phần đáng kể của +2.868 pp thì đây là phương pháp
khả thi; nếu không, hướng α thích ứng chết hẳn.

Cần lưu thêm đặc trưng thô trong `sweep_alpha.py` (~52 MB).

### 9.4 Thứ tự đề xuất

| # | Việc | Chi phí | Vì sao |
|---|---|---|---|
| 1 | **G** — tách `pseudo-unseen` | ~1 ngày | **Bắt buộc.** §3.7: best epoch đang chọn trên tập test, nên 78.5 chưa dùng được trong bảng so sánh |
| 2 | **H′** — α theo cụm | ~2 giờ | Kiểm được trên dữ liệu đã có; quyết định sống chết của hướng α thích ứng |
| 3 | **F** — re-ranking | ~2 giờ | Rẻ, không train lại, là hướng duy nhất **không** đi ngược mẫu hình "tăng thích nghi = tệ hơn" |
| 4 | **D, E** | ~4 giờ mỗi cái | Rẻ, độc lập |
| — | B′, C | — | Kỳ vọng thấp: cùng nhóm với weight decay (đã trơ) hoặc tăng thích nghi (đã hỏng) |

**Song song:** viết Khung A ngay. Nó không phụ thuộc kết quả nào ở trên, và nếu
1–4 đều âm thì nó là bài báo. Nếu có cái dương, nó thành phần phân tích của bài
phương pháp.

### 9.5 Điều cần trung thực trong mọi khung

- **§3.7** — con số 78.5 hưởng lợi từ việc chọn epoch trên tập test. Phải sửa
  (hướng G) hoặc nêu rõ trước khi so với công trình khác.
- **Một seed cho mỗi cấu hình là không đủ** — δ_min = 0.31 pp.
- **Oracle không phải trần** trừ khi đã kiểm chứng chia đôi.

---

## 10. Phụ lục: công cụ

| Công cụ | Vai trò |
|---|---|
| `scripts/verify_training.py` | 18 bất biến của vòng huấn luyện trên CLIP thật |
| `scripts/paired_test.py` | Kiểm định cặp trên AP từng query, bootstrap CI |
| `scripts/summarize_runs.py` | Bảng ablation + `--degradation` + phân tích epoch |
| `scripts/run_phase{0,1,2,3}.sh` | Điều phối, hỗ trợ `DRY=1` |
| `<log_dir>/<exp>/<run_id>/` | `config.json`, `run.log`, `train_steps.csv`, `metrics_epoch.csv`, `ap_best.npz` |
| `runs_summary.csv` | Một dòng mỗi run, ngoài `log_dir`, để so sánh chéo |

Xem thêm [`docs/param-leak-fix.md`](param-leak-fix.md) về năm chỗ rò rỉ tham số
đã sửa. **Lưu ý so sánh:** mọi kết quả trước bản vá đó không so trực tiếp được —
khi ấy thứ đang được huấn luyện là *CLIP fine-tuning một phần với prompt đóng
băng ở giá trị ngẫu nhiên*, chứ không phải prompt tuning.
