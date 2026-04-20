# Ke hoach implement `L_struct` cho HiCroPL-SBIR

## 1. Muc tieu

Muc tieu cua thay doi nay la them mot structural regularizer moi, `L_struct`, de ep khong gian embedding cua sketch va photo tuan theo cau truc ngu nghia da duoc CLIP text prototype dinh san. Khac voi cach tinh centroid theo batch de bi noisy, phien ban moi phai dung memory bank + EMA de tao target on dinh, sau do dung batch centroid co gradient de dua loss ve prompt parameters.

Ket qua mong doi:

- Cau truc embedding on dinh hon qua nhieu batch.
- Gradient van chay ve prompt learner va cac branch trainable.
- Loi so sanh cau truc phu hop voi geometry cua text prototypes, co loi cho zero-shot generalization.

## 2. Trang thai hien tai cua code

### 2.1. Pipeline hien tai

Hien tai, training pipeline da chay on dinh theo nhan xet sau:

- `src/model_hicropl.py` trich xuat feature cho photo, sketch, negative, va cac branch aug.
- `src/losses_hicropl.py` chi tinh loss chinh dua tren `cross_loss` + `cross_entropy`.
- `experiments/options.py` da co nhieu tham so prompt/adapters, nhung chua co cau hinh nao cho `L_struct` hoac memory bank.
- `src/hicropl.py` da co prompt flow bidirectional, nhung chua co structural memory bank, EMA centroid, hay precomputed text geometry cho toan bo seen classes.

### 2.2. Diem can giu nguyen

- Khong nen pha vo signature hien tai cua `CustomCLIP.forward()` neu co the tranh duoc.
- Khong nen thay doi logic prompt flow hien co neu chi can them structural regularization.
- Khong nen dua `L_struct` vao theo kieu tinh centroid batch thuan, vi do la van de da duoc xac dinh la noisy va gradient yeu.

### 2.3. Diem can bo sung

- Them config cho `lambda_struct`, `struct_warmup_epochs`, `struct_ema_momentum`, va cac tham so memory bank lien quan.
- Them memory bank cho photo/sketch centroids theo tung class.
- Precompute `S_text` cho toan bo seen classes.
- Them ham tinh `L_struct` voi hybrid gradient flow.
- Cap nhat test de bao ve luong moi.

## 3. Nguyen tac thiet ke

De dam bao step truoc luon la input cho step sau, luong can phai di theo chuoi sau:

1. Batch input tu dataloader cung cap `image`, `label`, `modality`, va cac bien the aug.
2. Model forward tao ra feature da normalize cho photo/sketch.
3. Feature batch va label duoc dung de cap nhat memory bank bang EMA trong `torch.no_grad()`.
4. Memory bank on dinh + batch centroid co gradient duoc dung de tinh similarity vector.
5. Similarity vector nay duoc so voi `S_text` da frozen de tao `L_struct`.
6. `L_struct` duoc cong vao total loss cung voi cac loss hien tai.
7. Total loss cap gradient nguoc ve prompt learner va branch trainable.

Co the hieu pipeline nay nhu mot chuoi dau vao-dau ra:

| Buoc | Dau ra cua buoc truoc | Dau vao cho buoc sau |
| --- | --- | --- |
| 1 | Batch image + label | Model forward |
| 2 | Photo/sketch features da normalize | Update memory bank |
| 3 | EMA centroids theo class | Batch centroid co gradient |
| 4 | Similarity vector voi EMA centroids | So sanh voi `S_text` |
| 5 | `L_struct` | `L_total` |
| 6 | `L_total` | Backprop + optimizer step |

## 4. Thiet ke implementation chi tiet

### Buoc 1. Them cau hinh cho structural loss

Cap nhat `experiments/options.py` de co day du tham so:

- `--lambda_struct`: trong so cua `L_struct`.
- `--struct_warmup_epochs`: so epoch warmup truoc khi bat dau tinh `L_struct`.
- `--struct_ema_momentum`: momentum cho EMA centroid.
- `--struct_bank_init_mode`: quy uoc cach khoi tao centroid ban dau, vi du random hoac theo zero vector.
- `--struct_include_sketch` / `--struct_include_photo` neu muon bat/tat rieng tung modality.

Output cua buoc nay la bo tham so day du de cac module khac co the tinh toan structural loss ma khong hard-code gia tri.

### Buoc 2. Dinh nghia interface cho structural memory bank

Them mot abstraction ro rang, uu tien dat trong `src/hicropl.py` hoac mot file helper moi neu muon tach logic:

- Luu `photo_centroids` va `sketch_centroids` theo kich thuoc `num_seen_classes x embed_dim`.
- Luu `class_initialized` hoac `class_counts` de biet class nao da co du lieu cap nhat.
- Luu `S_text` la matrix similarity co dinh giua tat ca seen-class text prototypes.
- Luu mapping giua label va index neu dataset label khong lien tuc.

Output cua buoc nay la mot memory bank object co the nhan feature + label va tra ve centroid on dinh cho cac buoc sau.

### Buoc 3. Precompute text geometry mot lan truoc khi train

Trong giai doan khoi tao model:

- Lay text prototypes cua toan bo seen classes tu CLIP frozen branch.
- L2 normalize toan bo text embeddings.
- Tinh similarity matrix `S_text = text_feats @ text_feats.T`.
- Freeze `S_text` va khong tinh lai trong training loop.

Vi tri phu hop de gan buoc nay la luc khoi tao model hoac luc tao structural bank.

Output cua buoc nay la `S_text` bat bien, lam target geometry cho tat ca cac batch ve sau.

### Buoc 4. Cap nhat memory bank sau moi forward

Sau khi `src/model_hicropl.py` tra ve feature cho photo va sketch:

- L2 normalize feature theo tung modality.
- Loc mau theo label trong batch.
- Tinh batch centroid cho tung class xuat hien trong batch.
- Cap nhat centroid EMA trong `torch.no_grad()`.
- Renormalize centroid ve unit sphere.

Quan trong: day chi la cap nhat state, khong phai phan cua loss. Ket qua cap nhat phai duoc luu lai de dung lam target on dinh cho buoc tinh loss.

Output cua buoc nay la memory bank da duoc lam moi, san sang cung cap EMA centroid cho structural matching.

### Buoc 5. Tinh batch centroid co gradient

Day la phan quan trong nhat de gradient quay ve prompt parameters:

- Voi moi class trong batch, tinh centroid batch hien tai tu feature co gradient.
- Khong detach centroid nay.
- L2 normalize centroid batch truoc khi tinh similarity.

Batch centroid co gradient nay se la query embedding cho buoc structural matching tiep theo.

Output cua buoc nay la centroid batch co gradient, lam input cho ham so sanh voi EMA memory bank.

### Buoc 6. Tinh similarity vector voi toan bo EMA centroids

Voi moi centroid batch co gradient:

- Tinh similarity voi tat ca EMA centroids trong memory bank.
- Loai bo class hien tai neu thiet ke loss yeu cau self-similarity khac 1, hoac giu lai neu muon buoc cau truc bao gom self-match.
- Su dung EMA centroids da detach lam target reference.

Song song, lay dong tuong ung trong `S_text` de lam target geometry.

Output cua buoc nay la hai vector can doi chieu:

- `sim_batch_to_memory`
- `sim_text_target`

### Buoc 7. Dinh nghia `L_struct`

Tinh `L_struct` bang MSE giua vector similarity thuc te va vector similarity text target:

- Tinh rieng cho photo.
- Tinh rieng cho sketch.
- Tong hop lai voi trong so tuy chon neu muon can bang hai modality.

Neu can on dinh hon, co the them warmup condition:

- Trong cac epoch dau, `L_struct = 0`.
- Sau warmup, bat dau cong vao total loss.

Output cua buoc nay la scalar `L_struct`, co gradient tu batch centroid ve prompt branch.

### Buoc 8. Tich hop vao total loss hien tai

Cap nhat `src/losses_hicropl.py` de total loss co dang:

`L_total = L_existing + lambda_struct * L_struct`

Trong do `L_existing` giu nguyen logic dang co cua project hien tai.

Neu can tranh lam thay doi code qua lon, co the:

- Tach `L_struct` thanh ham rieng.
- Goi no trong `training_step()` sau khi model forward xong.
- Hoac tra them mot field structural state tu model de loss module dung.

Output cua buoc nay la total loss moi, la dau vao truc tiep cho backprop va optimizer step.

### Buoc 9. Logging va quan ly debug

Can log it nhat cac thong tin sau:

- Gia tri `L_struct` theo step/epoch.
- So class da init trong memory bank.
- Do on dinh cua EMA centroid, vi du norm hoac drift.
- Muc dong gop cua `L_struct` so voi cac loss khac.

Output cua buoc nay la kha nang kiem tra xem structural loss co thuc su hoat dong hay khong, thay vi chi nhin vao metric cuoi cung.

### Buoc 10. Cap nhat test va xac thuc

Them/doi test de bao ve cac dieu kien sau:

- Memory bank cap nhat dung voi label.
- EMA centroid duoc renormalize ve unit sphere.
- `S_text` co shape bang so seen classes.
- `L_struct` bang 0 trong warmup, va lon hon 0 khi co batch hop le.
- Gradient van quay ve prompt parameters, khong bi detach truoc do.

Doan test hien tai co the chua khop hoan toan voi forward signature moi, nen can cap nhat theo output thuc te cua code hien tai.

## 5. Thu tu trien khai de tranh vo logic

De dam bao moi buoc chi dung output da hop le cua buoc truoc, nen lam theo thu tu nay:

1. Them config trong `experiments/options.py`.
2. Tao memory bank + `S_text` scaffold trong `src/hicropl.py` hoac helper moi.
3. Noi structural state vao `CustomCLIP` hoac `HiCroPL_SBIR` de co the cap nhat sau forward.
4. Implement `L_struct` trong `src/losses_hicropl.py` hoac mot module loss rieng.
5. Tich hop warmup va weighting vao `training_step()`.
6. Cap nhat test de assert shape, update rule, va gradient flow.

## 6. Muc tieu dau ra cua ban code moi

Sau khi hoan thanh, code moi phai dat duoc cac muc tieu sau:

- Co `L_struct` chay that, khong con la y tuong tren giay.
- Memory bank EMA on dinh hon per-batch centroid.
- Loss dung geometry cua text prototypes lam scaffold.
- Gradient van di qua batch centroid ve prompt learner.
- Training pipeline van giu duoc cac loss hien tai, chi them structural regularization theo cach co kiem soat.

## 7. Definition of done

Mot implementation duoc coi la xong khi:

- Training chay khong loi voi `lambda_struct = 0` va `lambda_struct > 0`.
- Memory bank duoc cap nhat dung trong toi thieu mot epoch.
- `L_struct` co gia tri on dinh va co dong gop vao total loss.
- Test co the xac nhan gradient flow van thong suot.
- Khong co thay doi khong can thiet trong prompt flow hien co.
