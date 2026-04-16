# Bi-Knowledge Flow Design: Additive Injection (Shallow Phase)

## Status Update (2026-04-16)

This document started as design-only, but core additive ideas have been integrated in code.

- The design intent in this file is still valid.
- Some wording that implies "pre-implementation only" should be interpreted as historical.
- For post-fix runtime notes (prompt-drift and update-noise fixes), also read:
  - `docs/prompt_update_noise_fixes.md`

## 1. Muc tieu

Tai lieu nay mo ta mot huong thiet ke de sua shallow-phase bi-knowledge flow theo nguyen tac:

- Khong split visual tokens theo 2+2.
- Giu nguyen toan bo co che text->visual hien tai (full-token update).
- Them thong tin sketch nhu mot residual contribution duoc gate.
- Gate khoi tao nho de hanh vi ban dau gan nhu baseline.

Huong nay nham giam rui ro mat text guidance o shallow layers, dong thoi van mo duong cho cross-modal transfer neu co loi cho retrieval.

## 2. Van de can giai quyet

Trong cac variant co split token, mot phan visual tokens cua photo khong nhan text guidance truc tiep o shallow phase. Dieu nay co the:

- Lam giam semantic alignment text-photo som.
- Tao dao dong toi uu o giai doan dau.
- Lam mo nhat diem tuong dong voi baseline on dinh.

Y tuong Additive Injection giai quyet bang cach cho tat ca visual tokens van nhan text guidance, sau do chi bo sung sketch signal theo residual.

## 3. Nguyen ly cot loi

Shallow phase cho nhanh photo duoc tach thanh 2 thanh phan cong huong:

1. Thanh phan chinh (core flow):
- Full text->visual update tren toan bo visual tokens.
- Day la duong giu nguyen 100% tinh chat baseline.

2. Thanh phan bo sung (residual flow):
- Sketch-derived contribution tinh tren cung query visual tokens.
- Duoc nhan voi mot he so gate alpha co hoc.

Tong cap nhat shallow visual prompt:

- Visual_updated = Visual_from_text + alpha * Visual_from_sketch

Trong do alpha duoc khoi tao rat nho de he thong ban dau gan baseline.

## 4. Kien truc de xuat cho shallow phase

### 4.1 Text-guided branch (giu nguyen)

- Query: toan bo photo shallow visual tokens.
- Key/Value: text proxies cua photo branch.
- Output: updated_by_text.

### 4.2 Sketch residual branch (bo sung)

- Query: cung toan bo photo shallow visual tokens.
- Key/Value: sketch shallow proxies da chuan hoa shape.
- Output: sketch_contribution.

### 4.3 Gated fusion

- updated_visual_prompts = updated_by_text + gate_alpha * sketch_contribution

Y nghia:

- Neu gate_alpha gan 0: hanh vi gan baseline.
- Neu gate_alpha hoc tang: cho phep sketch signal tham gia manh hon.

## 5. Gate alpha

De xuat:

- gate_alpha la tham so hoc duoc (learnable scalar).
- Khoi tao nho, vi du 0.01.

Tac dung:

- Warm start an toan.
- Tranh shock optimization o giai doan dau.
- Cho model tu quyet dinh muc do su dung cross-knowledge.

Mo rong co the can nhac:

- 1 gate toan cuc.
- 1 gate theo layer shallow.
- 1 gate theo modality direction.

Ban dau nen dung 1 gate scalar de de debug va ablation.

## 6. Yeu cau shape va du lieu

Can dam bao cac tensor dong shape khi fusion:

- updated_by_text va sketch_contribution cung kich thuoc:
  - [cross_layer, n_ctx_photo, v_dim]

- external sketch proxies can duoc:
  - dua ve dung device
  - dua ve dung dtype
  - dua ve dung truong hop theo layer

Neu shape sketch proxy khong khop, can co buoc adapt ro rang (repeat/truncate/projection) va ghi log.

## 7. Tich hop voi luong hien tai

Huong nay chi tac dong shallow phase update cua photo prompts.

Cac luong khac giu nguyen:

- Text encoder flow
- Visual encoder flow
- Deep phase flows da co
- Distill/fixed embedding flow
- Optimizer grouping

Nghia la thay doi co tinh cuc bo, de danh gia tac dong thuan cua shallow additive injection.

## 8. Uu diem ky vong

- Bao toan baseline semantics o shallow phase.
- Tranh mat thong tin text tren bat ky token nao.
- Cross-modal sketch signal duoc bo sung mem theo residual.
- On dinh toi uu tot hon so voi split cứng o giai doan dau.

## 9. Rui ro va cach giam

### 9.1 Rui ro

- Sketch residual co the van qua manh neu gate hoc nhanh.
- Co the tao noise neu proxy sketch chat luong kem.
- Co the tang do nhay loss o giai doan warm-up.

### 9.2 Giam thieu

- Khoi tao gate_alpha rat nho.
- Co the rang buoc gate_alpha >= 0 bang softplus/exp parameterization neu can.
- Theo doi norm cua sketch contribution theo epoch.
- Theo doi gate_alpha progression tren TensorBoard.

## 10. Ke hoach ablation de xac nhan y tuong

De xuat thu nghiem toi thieu:

1. Baseline hien tai (khong additive sketch residual)
2. Additive residual + learnable gate (init 0.01)
3. Additive residual + gate fixed (0.01) de tach tac dong cua hoc gate
4. Additive residual + gate theo layer (neu can)

Metrics can theo doi:

- mAP
- P@K
- top1/top5 (neu fine-grained)
- train stability (loss variance, gradient norm)
- gate_alpha trajectory

## 11. Tieu chi chap nhan

Y tuong duoc coi la thanh cong neu:

- Khong lam giam baseline metric trong warm-up va early epochs.
- Co xu huong cai thien retrieval metric o mid/late training.
- Gate alpha tang hoac giam co y nghia (khong diverge bat thuong).
- Khong gay nan/inf va khong lam vo shape invariants.

## 12. Pham vi tai lieu

Tai lieu nay la design doc cho huong Additive Injection.

- Khong bao gom patch code chi tiet.
- Khong thay doi API public cua model trong tai lieu nay.
- Dung lam co so truoc khi code implementation.
