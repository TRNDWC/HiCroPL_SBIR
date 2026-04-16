# Bi-Knowledge Flow Implementation Notes

## Status Update (2026-04-16)

This document records an earlier implementation/proposal stage.

- Keep this file as historical context for design evolution.
- Do not treat every behavior here as current runtime behavior.
- For current behavior and fixes, read:
  - `docs/bi_knowledge_flow_additive_injection_design.md`
  - `docs/prompt_update_noise_fixes.md`
- Final authority remains current code in `src/hicropl.py` and `src/model_hicropl.py`.

## 1. Muc tieu

Tai lieu nay mo ta luong bi-knowledge flow hien tai da duoc implement trong CrossModalPromptLearner, voi cac nguyen tac:

- Giu nguyen toan bo luong hien tai dang co.
- Bo sung luong moi giua photo va sketch theo dung mo ta.
- Mo ta dung state code hien tai, khong phai proposal nua.

## 2. Trang thai hien tai (baseline flow)

He thong hien tai da co 2 cap nhat noi bo theo tung modality:

- Nhanh photo:
  - Tang nong: text_photo update photo (T->I)
  - Tang sau: photo update text_photo (I->T)

- Nhanh sketch:
  - Tang nong: text_sketch update sketch (T->I)
  - Tang sau: sketch update text_sketch (I->T)

Trong de xuat moi, cac luong tren duoc giu nguyen va duoc coi la baseline.

Ngoai ra, code hien tai da co the hien rong hon so voi baseline:

- Prompt learner duoc tach ro theo photo va sketch.
- Moi nhanh co parameter, mapper, proxy, va cache rieng.
- Cac luong bo sung bi-flow duoc chen vao tren prompt visual theo branch.

## 3. Bi-flow da implement

He thong hien tai da bo sung 2 luong cheo modality:

1. Tang nong (cross_layer): sketch update photo
2. Tang sau (deep layers): photo update sketch

Y tuong cot loi:

- O tang nong, prompt photo moi layer co 4 tokens se duoc chia 2 khoi:
  - Khoi A: 2 token dau cua photo, chi nhan update tu text.
  - Khoi B: 2 token sau cua photo, chi nhan update tu sketch.

- O tang sau, bo sung cap nhat theo chuoi:
  - text_photo update photo
  - photo update sketch

Nghia la tang sau cua sketch se nhan them thong tin da duoc tinh chinh boi text_photo thong qua photo.

## 4. Dinh nghia token-level cho tang nong

Gia su moi layer nong co n_ctx = 4 token cho photo:

- photo_tokens[layer] = [p0, p1, p2, p3]
- Tach thanh 2 khoi:
  - photo_head = [p0, p1]
  - photo_tail = [p2, p3]

### 4.1 Khoi A (text -> photo_head)

- Query: photo_head
- Key/Value: text-side proxies (nhu luong T->I hien tai)
- Muc dich: giu nguyen co che text guidance cho mot nua prompt photo.

### 4.2 Khoi B (sketch -> photo_tail)

- Query: photo_tail
- Key/Value: sketch-side proxies (hoac sketch prompt summary o tang nong)
- Muc dich: dua thong tin sketch vao nua con lai cua prompt photo o cung layer.

### 4.3 Hop nhat sau mapper

- photo_updated[layer] = concat(updated_head, updated_tail)
- Thu tu token duoc bao toan: [head(2), tail(2)]

Dieu nay tao ra hai khong gian chuc nang trong cung mot layer nong cua photo:

- Nua dau nghieng ve text semantics
- Nua sau nghieng ve cross-modal sketch signal

## 5. Dinh nghia luong tang sau (photo -> sketch)

Ngoai luong I->T noi bo tung nhanh, bo sung luong cheo:

- text_photo update photo (giu nguyen luong hien tai)
- photo update sketch (luong moi)

### 5.1 Chuoi cap nhat de xuat

O moi deep layer i (i >= cross_layer):

1. Tinh photo_i da update boi text_photo theo luong cu.
2. Dung photo_i (hoac proxy cua photo_i) lam Key/Value.
3. Dung sketch_i lam Query de cap nhat sketch_i.

Ket qua:

- sketch deep prompts duoc huong bo sung boi ngu canh photo da qua text-guidance.
- Tao chuoi tri thuc: text_photo -> photo -> sketch.

## 6. Luong du lieu thuc te trong code

### 6.1 Giai doan tang nong

- Photo self-flow cu: text_photo -> photo
- Sketch self-flow cu: text_sketch -> sketch
- Them flow moi: sketch -> photo (chi tren half-tail tokens cua photo)

Tong ket tang nong photo:

- head(2 tokens): nhan text signal
- tail(2 tokens): nhan sketch signal

### 6.2 Giai doan tang sau

- Giu flow cu:
  - photo -> text_photo
  - sketch -> text_sketch
- Them flow moi:
  - photo -> sketch

Trong do photo dung de update sketch sau khi photo da nhan text_photo guidance.

## 7. Cac phuong an tao proxy cho luong cheo

De xuat khong rang buoc 1 cach duy nhat, co the chon 1 trong 3 phuong an:

1. Proxy pooling dong nhat voi flow cu:
- Tao proxy sketch tu sketch prompts o tang nong.
- Tao proxy photo tu photo prompts o tang sau.

2. Mean pooling token-level:
- Don gian hon, dung mean theo token dimension.

3. Hybrid:
- Tang nong dung pooling hoc duoc.
- Tang sau dung mean pooling de giam do phuc tap.

Nen uu tien phuong an 1 de dong nhat voi thiet ke HicroPL.

## 8. Rang buoc hinh hoc va shape

Dieu kien can:

- n_ctx cua nhanh photo phai >= 4 de chia 2+2 nhu yeu cau.
- Neu n_ctx != 4, can dinh nghia lai quy tac chia (vi du floor/ceil).

Mapper shape muc tieu:

- Block A query size: 2 tokens/layer x cross_layer layers
- Block B query size: 2 tokens/layer x cross_layer layers

Key/Value cua moi block duoc xay tu proxy tuong ung (text hoac sketch).

## 9. Thu tu uu tien cap nhat trong mot step

De tranh xung dot in-place update, can quy dinh thu tu tinh toan ro rang:

1. Snapshot prompt dau vao theo layer (neu can).
2. Tinh block A va block B tu cung mot trang thai dau vao.
3. Ghep ket qua va commit vao photo layer.
4. Chay tiep deep-stage update voi thu tu da dinh nghia.

Neu khong lam ro thu tu, co the xay ra phu thuoc cap nhat ngoai y muon.

## 10. Anh huong ky vong

### 10.1 Loi ich

- Tang kha nang dong bo photo-sketch o layer prompt som.
- Tao phan cong chuc nang token-level ro rang trong prompt photo.
- Truyen tri thuc text_photo gian tiep sang sketch qua photo o tang sau.

### 10.2 Rui ro

- Tang so luong mapper operation, de bat on train neu LR cao.
- Co the lam photo bi over-conditioned boi sketch neu khong can bang.
- Can theo doi sat norm prompt va metric retrieval de tranh collapse.

## 11. De xuat hyper-parameter gate (khuyen nghi)

Du day la de xuat design-only, nen du phong cac he so gate cho giai doan implement:

- alpha_head_text: do manh block A
- beta_tail_sketch: do manh block B
- gamma_photo_to_sketch_deep: do manh flow photo->sketch o tang sau

Cac gate nay giup bat/tat hoac anneal tung flow de ablation de dang hon.

## 12. Ke hoach ablation sau nay

Khi implement, nen danh gia theo thu tu:

1. Baseline hien tai (khong flow cheo)
2. + Tang nong sketch->photo (2+2 split)
3. + Tang sau photo->sketch
4. + Day du ca 2 flow

Bao cao toi thieu:

- mAP
- P@K
- Top1/Top5 (fine-grained neu co)
- Do on dinh train (loss curve, norm prompt)

## 13. Pham vi tai lieu nay

Tai lieu nay mo ta state hien tai da implement va cac y tuong dau gia tiep theo.

- Khong bao gom code
- Khong thay doi API hien tai
- Khong thay doi behavior luc nay

No duoc dung lam dac ta cho state hien tai va co the lam co so cho implementation tiep theo.
