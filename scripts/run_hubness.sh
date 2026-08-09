#!/usr/bin/env bash
# Train lại từ đầu rồi chạy chẩn đoán + sửa hubness, trên NHIỀU SEED.
#
# Vì sao nhiều seed: δ_min = 0.31 pp là dao động giữa các seed (docs §2). Mọi
# kết luận từ MỘT seed đều không phân biệt được với nhiễu. Đây là sai lầm đã đảo
# ngược kết luận ba lần trong dự án này.
#
# Vì sao train lại: Giai đoạn 2/3 chạy --save_top_k=0 nên không có checkpoint.
# Nếu saved_models/<exp> đã có .ckpt (vd. từ run_alpha_sweep.sh / run_qe_confirm.sh)
# thì script DÙNG LẠI, không train đè.
#
# Chi phí: ~40 phút train mỗi seed + ~5 phút phân tích. 3 seed ≈ 2 giờ.
# Checkpoint ~2.5GB mỗi seed.
#
# Chạy:
#   bash scripts/run_hubness.sh
#   SEEDS="1 2 3 4 5" bash scripts/run_hubness.sh
#   DRY=1 bash scripts/run_hubness.sh          # in lệnh, không chạy
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"1 2 3"}
EPOCHS=${EPOCHS:-8}
TAG=${TAG:-p4}
QE_K=${QE_K:-20}                 # qe_k tốt nhất từ run_qe_confirm.sh
BRANCHES=${BRANCHES:-"mixed frozen"}

# Đúng cấu hình nền đã cho 78.4 — KHÔNG đổi gì ở đây, nếu không thì con số
# không so được với mọi kết quả trước.
COMMON=(
  --dataset=sketchy_ext --data_dir=../data/Sketchy
  --gpt_text_file=gpt_file/sketchy_ext.json
  --n_ctx=2 --prompt_depth=12 --language_depth=1 --cross_layer=6
  --disable_cross_exchange --lambda_cross_modal=1.0 --lambda_ce=1.0
  --weight_decay=0 --clip_LN_lr=1e-6 --prompt_lr=1e-4
  --batch_size=128 --workers=4 --test_batch_size=1024
  --disable_augmentation --no_resume --save_top_k=1
)

for s in $SEEDS; do
  EXP="${TAG}_alphasweep_s${s}"
  if ls saved_models/"$EXP"/*.ckpt >/dev/null 2>&1; then
    echo "# seed $s: đã có checkpoint, bỏ qua train"
  else
    echo; echo "############ Train seed $s (từ đầu) ############"
    if [ "${DRY:-0}" = "1" ]; then
      echo "+ python -u -m experiments.hicropl_prompt ${COMMON[*]} --exp_name=$EXP --seed=$s --epochs=$EPOCHS"
    else
      python -u -m experiments.hicropl_prompt "${COMMON[@]}" \
        --exp_name="$EXP" --seed="$s" --epochs="$EPOCHS"
    fi
  fi

  # KHÔNG dùng `ls | sort | tail -1`: TensorBoardLogger tạo tb_logs/<exp>/version_N/
  # nằm cạnh thư mục run, và 'v' (0x76) > '2' (0x32) nên version_0 luôn thắng.
  # Lọc theo config.json — chỉ thư mục run mới có file đó.
  RD=$(find "tb_logs/$EXP" -mindepth 2 -maxdepth 2 -name config.json 2>/dev/null \
       | xargs -r -n1 dirname | sort | tail -1 || true)
  if [ "${DRY:-0}" = "1" ]; then
    echo "+ python scripts/hubness.py --run_dir <run_dir> --hub_branch $BRANCHES --qe_k $QE_K --save_ap"
    continue
  fi
  [ -n "$RD" ] || { echo "!! không thấy run_dir cho $EXP"; continue; }

  echo; echo "############ Hubness trên seed $s ############"
  python scripts/hubness.py --run_dir "$RD" --hub_branch $BRANCHES \
    --qe_k "$QE_K" --save_ap
done

[ "${DRY:-0}" = "1" ] && exit 0

echo; echo "############ Tổng hợp qua các seed ############"
python - "$TAG" $SEEDS <<'PY'
import csv, glob, json, os, statistics as st, sys

tag, seeds = sys.argv[1], sys.argv[2:]
DMIN = 0.31                                  # pp — ngưỡng phân giải seed (docs §2)

diags, per, bases, missing = [], {}, [], []
for s in seeds:
    # Một exp_name có thể có NHIỀU run_dir. Lấy cái MỚI NHẤT và in ra để kiểm
    # được — nếu không sẽ đọc phải CSV của lần chạy cũ.
    hits = sorted(glob.glob(f'tb_logs/{tag}_alphasweep_s{s}/*/hubness.csv'))
    if not hits:
        missing.append(s); continue
    p = hits[-1]
    rows = list(csv.DictReader(open(p, encoding='utf-8')))
    b = next((float(r['mAP']) for r in rows if r['method'] == 'baseline'), None)
    if b is None:
        missing.append(s); continue
    print(f'  seed {s}: nền {100*b:.3f}  <- {p}')
    bases.append(b)
    for r in rows:
        if r['method'] == 'baseline':
            continue
        per.setdefault((r['method'], r['param']), []).append(float(r['mAP']) - b)
    dj = os.path.splitext(p)[0] + '_diag.json'
    if os.path.exists(dj):
        diags.append(json.load(open(dj, encoding='utf-8')))

if missing:
    print(f'  !! thiếu kết quả cho seed: {missing} — bảng dưới KHÔNG đủ seed')
if not per:
    raise SystemExit('Không đọc được hubness.csv nào.')
print(f'\nnền: {100*st.fmean(bases):.3f}' +
      (f' ± {100*st.stdev(bases):.3f} pp (n={len(bases)} seed)' if len(bases) > 1
       else f' (n=1 — KHÔNG kết luận được gì)'))

# ---- A. chẩn đoán: đây mới là chỗ quyết định có câu chuyện hay không ----
if diags:
    def m(k):
        v = [d[k] for d in diags if k in d]
        return (st.fmean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else (float('nan'), 0.0)
    print(f'\n{"="*66}\nA. CHẨN ĐOÁN (trung bình qua {len(diags)} seed)\n{"="*66}')
    print(f'{"đại lượng":<34}{"trung bình":>13}{"std":>10}')
    print('-' * 57)
    for k, lbl in (('skew_mixed', 'skew N_k — mixed (chéo miền)'),
                   ('skew_photo2photo', 'skew N_k — photo→photo (cùng miền)'),
                   ('skew_frozen', 'skew N_k — frozen'),
                   ('skew_prompted', 'skew N_k — prompted'),
                   ('r_class_mAP_vs_badhub', 'r(mAP lớp, bị hub sai lớp nuốt)'),
                   ('hub_share', 'tỉ lệ ô top-k là hub (đều = 0.01)')):
        a, sd = m(k)
        print(f'{lbl:<34}{a:>13.3f}{sd:>10.3f}')
    sx, _ = m('skew_mixed'); sp, _ = m('skew_photo2photo'); r, _ = m('r_class_mAP_vs_badhub')
    print()
    print(f'  [1] hubness do KHOẢNG CÁCH MIỀN?  skew(chéo)/skew(cùng miền) = {sx/sp:.2f}x'
          f'  -> {"CÓ" if sx > 1.3*sp else "KHÔNG (chỉ là hiệu ứng nhiều chiều)"}')
    print(f'  [2] lớp yếu bị hub nuốt?          r = {r:+.3f}'
          f'  -> {"CÓ" if r < -0.3 else "KHÔNG"}')

# ---- B. các phép sửa ----
print(f'\n{"="*66}\nB. CÁC PHÉP SỬA\n{"="*66}')
PLAC = [d['placebo_threshold'] for d in diags if 'placebo_threshold' in d]
plac = 100 * max(PLAC) if PLAC else 0.0
TIE = [d['tie_frac'] for d in diags if 'tie_frac' in d]
if TIE:
    print(f'hoà điểm trong top-K: {100*st.fmean(TIE):.2f}% truy vấn')
print(f'ngưỡng giả dược (lệch cột ngẫu nhiên, tệ nhất qua seed): {plac:.3f} pp')
BAR = max(DMIN, plac)
print(f'-> phải vượt max(δ_min, giả dược) = {BAR:.3f} pp mới có nghĩa\n')

print(f'{"phương pháp":<22}{"tham số":<26}{"Δ tb":>9}{"std":>8}{"min":>8}{"n":>4}')
print('-' * 77)
for (meth, param), v in sorted(per.items(), key=lambda kv: -st.fmean(kv[1])):
    if meth.startswith('giả dược'):
        continue                      # là NGƯỠNG, không phải phương pháp
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f'{meth:<22}{param:<26}{100*st.fmean(v):>+9.3f}{100*sd:>8.3f}'
          f'{100*min(v):>+8.3f}{len(v):>4}')

real = {k: v for k, v in per.items() if not k[0].startswith('giả dược')}
(bm, bp), bv = max(real.items(), key=lambda kv: st.fmean(kv[1]))
ok = len(bv) > 1 and 100 * min(bv) > BAR
print(f'\nTốt nhất: {bm} ({bp}) -> {100*st.fmean(bv):+.3f} pp')
if len(bv) > 1:
    print(f'  khoảng qua seed [{100*min(bv):+.3f}, {100*max(bv):+.3f}] pp')
    print(f'  {"ĐẠT" if ok else "CHƯA ĐẠT"}: mọi seed đều {">" if ok else "<="} {BAR:.3f} pp')

# ---- C. cơ chế: nhóm yếu có tăng nhiều hơn không ----
if diags and 'gain_weak' in diags[0]:
    gw = [d['gain_weak'] for d in diags if 'gain_weak' in d]
    gr = [d['gain_rest'] for d in diags if 'gain_rest' in d]
    print(f'\n{"="*66}\nC. CƠ CHẾ (phép sửa tốt nhất mỗi seed)\n{"="*66}')
    print(f'  nhóm 7 lớp yếu nhất : {100*st.fmean(gw):+.3f} pp')
    print(f'  nhóm còn lại        : {100*st.fmean(gr):+.3f} pp')
    d = st.fmean(gw) - st.fmean(gr)
    print(f'  chênh               : {100*d:+.3f} pp -> '
          f'{"CƠ CHẾ ĐƯỢC XÁC NHẬN" if 100*d > BAR else "KHÔNG xác nhận — chỉ là hậu xử lý chung"}')
PY

cat <<'EOF'

################################################################
Ba câu hỏi, trả lời độc lập nhau:

  A[1] skew chéo miền >> cùng miền
       -> hubness sinh ra từ KHOẢNG CÁCH MIỀN, không phải từ số
          chiều. Đây là phát biểu đáng viết, và nó ĐÚNG hay SAI
          không phụ thuộc phép sửa có ăn điểm hay không.

  A[2] r âm mạnh
       -> door/saw/window yếu vì bị hub nuốt, không phải vì đặc
          trưng tồi. Đây là chẩn đoán CẠNH TRANH với PuXIM ("nền
          gây nhiễu") chứ không trùng.

  C    nhóm yếu tăng nhiều hơn nhóm còn lại
       -> phép sửa tác động đúng chỗ giả thuyết dự đoán.

Cả ba dương  -> có đóng góp: chẩn đoán + phép sửa nhắm đúng.
Chỉ B dương  -> chỉ là thủ thuật hậu xử lý, báo cáo thành một dòng.
A âm         -> giả thuyết sai, ghi vào danh sách hướng âm (docs §6)
                và dừng. Đây là kết quả hợp lệ, không phải thất bại.

Kiểm chặt chênh lệch nhỏ (--save_ap đã ghi vector AP):
    python scripts/paired_test.py \
      tb_logs/p4_alphasweep_s1/<run>/ap_hub_baseline.npz \
      tb_logs/p4_alphasweep_s1/<run>/ap_hub_csls.npz

Dọn checkpoint khi xong (~2.5GB mỗi seed):
    rm -rf saved_models/p4_alphasweep_s*
################################################################
EOF
