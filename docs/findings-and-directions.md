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

Hệ quả: hướng đi phải là một **ràng buộc có mục tiêu** giữ đúng cấu trúc ngữ
nghĩa của CLIP — không phải bỏ bớt loss, không phải regularization chung chung,
không phải thêm/bớt capacity.

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

---

## 4. Chẩn đoán tổng hợp

Ba mũi can thiệp, ba kết quả loại trừ lẫn nhau:

| Can thiệp | Kết quả |
|---|---|
| **Bỏ** một số hạng loss | tệ hơn 0.83 pp |
| **Thêm** regularization tổng quát | trơ (0.014 pp) |
| **Thêm** capacity | tệ hơn nhiều (suy giảm ×9.8) |

**[SUY]** Suy giảm **không đến từ một khuyết tật cụ thể nào** mà là hệ quả nội
tại của việc khớp phân phối huấn luyện. Do đó lời giải phải là một cơ chế **giữ
lại thứ cần giữ** — cấu trúc tương đồng giữa các lớp mà CLIP đã học — chứ không
phải điều chỉnh cường độ học.

**[SUY]** Hiện tại, thứ **duy nhất** đang giữ tính tổng quát là **residual mix**:

```python
feat = norm( norm(prompted) + frozen )
```

Nó ép cứng một nửa đặc trưng cuối là CLIP đóng băng. Hiệu quả, nhưng:

- **không học được** — tỉ lệ 1:1 là hằng số áp đặt
- **đối xứng** giữa hai modality, trong khi CLIP mạnh trên ảnh và yếu trên
  sketch, nên nhánh sketch bị neo vào chính phần đặc trưng kém chất lượng nhất
- **chặn trần** của mọi cải tiến ở cơ chế prompt — đây là lý do baseline 169,984
  params và cross exchange 46.3M params đều hội tụ về ~78.3

---

## 5. Hướng phát triển

Mỗi hướng ghi rõ **cơ sở** (bằng chứng nào dẫn tới nó), **cơ chế**, **chi phí**,
và **cách bác bỏ** — nếu một đề xuất không nêu được cách nó có thể sai thì nó
chưa phải giả thuyết khoa học.

### A. Trọng số trộn residual học được theo modality — *ưu tiên cao*

**Cơ sở.** §4: residual mix là cơ chế giữ tổng quát duy nhất, đang cố định và
đối xứng. §3.5: mọi cách thêm capacity vào nhánh prompted đều thất bại, nên đòn
bẩy còn lại nằm ở **tỉ lệ tin nhánh đó**.

**Cơ chế.**

```
feat = norm( a · prompted + (1−a) · frozen ),   a = sigmoid(θ)
```

`θ` riêng cho photo và sketch. **Hai tham số cho cả mô hình.**

**Tính chất then chốt:** `θ = 0 → a = 0.5 →` trùng đúng `norm(u + f)`, đã kiểm
bằng số (sai khác `0.000e+00`). Nên mọi khác biệt quan sát được **chắc chắn** do
`a` học ra, không do đổi công thức.

**Giá trị lớn nhất không nằm ở mAP.** Quỹ đạo của `a` là **lời khai của chính mô
hình** về việc nên tin nhánh prompted bao nhiêu:

| Quan sát | Kết luận |
|---|---|
| `a_sketch ≠ a_photo` | bất đối xứng modality có thật |
| `a → 0` | nhánh prompted **gây hại**, residual mix cố định đang che giấu |
| `a` đứng yên 0.5 | gradient không tới được — kiểm `mix_alpha_lr` |

Trường hợp `a → 0` là kết quả mạnh **dù mAP không đổi**: nó thống nhất toàn bộ
chuỗi kết quả âm (cross exchange trung tính, capacity có hại, bỏ loss có hại)
thành một câu chuyện duy nhất.

**Trạng thái.** Đã cài đặt (`--learn_mix_alpha`), `scripts/run_phase3.sh`, chưa
chạy.

**Cách bác bỏ.** Nếu `a` ở lại quanh 0.5 với nhiều `mix_alpha_lr` khác nhau thì
nhánh prompted đang đóng góp thật và hướng B rủi ro hơn nhiều.

### B. Thay residual mix bằng ràng buộc quan hệ tường minh — *trần cao nhất, rủi ro cao nhất*

**Cơ sở.** §4: cần cơ chế giữ đúng thứ cần giữ. §3.2 + §3.5: mọi cải tiến bên
trong cơ chế prompt đều bị residual mix dập tắt.

**Cơ chế.** Giữ **cấu trúc quan hệ** thay vì **giá trị tuyệt đối**:

```
L_rel = ‖ sim(F_prompted) − sim(F_frozen) ‖²_F
```

trên ma trận tương đồng trong batch. Rồi **giảm dần trọng số nhánh frozen về 0**,
để đặc trưng cuối là nhánh prompted thuần.

**Vì sao có thể hiệu quả [GIẢ].** Relational distillation cho phép đặc trưng dịch
chuyển tự do **miễn là giữ hình học tương đối giữa các lớp** — chính xác là thứ
quyết định tổng quát hoá zero-shot. Prompt được giải phóng để thu hẹp khoảng cách
miền thay vì chỉ tinh chỉnh quanh một nền cố định.

**Lộ trình an toàn.** `a` cố định 1:1 → `a` học được (A) → `a` + `L_rel` →
anneal `(1−a)` về 0 → bỏ hẳn nhánh frozen.

**Cách bác bỏ.** Nếu bỏ nhánh frozen làm mAP sập dù `L_rel` mạnh, thì residual
mix không chỉ là thủ thuật mà là thành phần không thể thay thế.

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

**Rủi ro [ĐO gián tiếp].** §3.5 cho thấy thêm capacity làm overfit nặng hơn. Prompt
điều kiện instance là thêm capacity — nên **phải đi kèm A hoặc B**, không chạy
độc lập.

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
| Thêm weight decay để chống overfit | §3.4 — lệch 0.014 pp, dưới δ_min 22 lần |
| Bỏ L4 để giảm overfit | §3.3 — best tệ hơn 0.83 pp, KTC `[−0.93, −0.71]` |
| Giữ static cross exchange làm base | §3.2 — trung tính ở giá 272× tham số, và làm chậm mọi vòng lặp |
| Chạy 60 epoch | §3.1 — best epoch 2–3; 8 epoch là đủ, tiết kiệm 7.5× |
| Kết luận từ chênh lệch mAP < 0.31 pp | §2.1 — đó là δ_min |
| So hai cấu hình bằng 1 seed | §2.1 — std seed 0.32 pp |

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

### 7.3 Quỹ đạo `a` (hướng A)

Đã cài đặt, chưa chạy.

---

## 8. Thứ tự đề xuất

1. **`--eval_frozen_only`** (~10 phút) — quyết định mọi thứ còn lại (§7.1)
2. **Hướng A** (~4 giờ) — 2 tham số, quỹ đạo `a` cho câu trả lời trực tiếp
3. Rẽ nhánh theo kết quả:
   - `a → 0` hoặc frozen-only cao → **hướng B**
   - `a` quanh 0.5 → **hướng C** kèm A, và **hướng D**
4. **E, F** đi kèm bất kể nhánh nào

---

## 9. Phụ lục: công cụ

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
