#!/usr/bin/env bash
# Xác nhận αQE trên NHIỀU SEED.
#
# Kết quả một checkpoint: 78.500 -> 81.756 (+3.256 pp), đỉnh tại qe_k=20.
# Đó là cải thiện lớn nhất toàn dự án, nhưng mới trên MỘT seed. δ_min = 0.31 pp
# là dao động seed, nên trước khi công bố phải biết:
#   - mức tăng có ổn định giữa các seed không
#   - qe_k tối ưu có giống nhau không, hay phải tune mỗi lần
#
# αQE là hậu xử lý xác định, nên toàn bộ dao động đến từ checkpoint khác nhau.
#
# Chạy:
#   bash scripts/run_qe_confirm.sh
#   SEEDS="1 2 3 4 5" bash scripts/run_qe_confirm.sh
#   DRY=1 bash scripts/run_qe_confirm.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"1 2 3"}
EPOCHS=${EPOCHS:-8}
TAG=${TAG:-p4}
QE_K=${QE_K:-"0 5 10 20 40"}

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
  # Seed 1 đã chạy ở bước quét α — dùng lại, đừng train lại.
  if [ -d "saved_models/$EXP" ] && ls saved_models/"$EXP"/*.ckpt >/dev/null 2>&1; then
    echo "# đã có checkpoint cho seed $s, bỏ qua train"
  else
    echo; echo "############ Train seed $s ############"
    if [ "${DRY:-0}" = "1" ]; then
      echo "+ python -u -m experiments.hicropl_prompt ... --exp_name=$EXP --seed=$s"
    else
      python -u -m experiments.hicropl_prompt "${COMMON[@]}" \
        --exp_name="$EXP" --seed="$s" --epochs="$EPOCHS"
    fi
  fi

  RD=$(find "tb_logs/$EXP" -mindepth 2 -maxdepth 2 -name config.json 2>/dev/null \
       | xargs -r -n1 dirname | sort | tail -1 || true)
  [ -n "$RD" ] || { echo "!! không thấy run_dir cho $EXP"; continue; }
  echo; echo "############ αQE trên seed $s ############"
  if [ "${DRY:-0}" = "1" ]; then
    echo "+ python scripts/improve_eval.py --run_dir $RD --skip_cluster --qe_k $QE_K"
  else
    python scripts/improve_eval.py --run_dir "$RD" --skip_cluster --qe_k $QE_K
  fi
done

[ "${DRY:-0}" = "1" ] && exit 0

echo; echo "############ Tổng hợp qua các seed ############"
python - "$TAG" $SEEDS <<'PY'
import csv, glob, os, statistics as st, sys
tag, seeds = sys.argv[1], sys.argv[2:]
per_k, bases = {}, []
missing = []
for s in seeds:
    # Một exp_name có thể có NHIỀU run_dir (mỗi lần chạy một cái). Lấy cái MỚI
    # NHẤT theo tên thư mục timestamp, và in ra để kiểm được — bản trước lấy
    # kết quả đầu tiên glob trả về, nên có thể đọc phải CSV cũ của lần chạy khác.
    hits = sorted(glob.glob(f'tb_logs/{tag}_alphasweep_s{s}/*/improve_eval.csv'))
    if not hits:
        missing.append(s)
        continue
    p = hits[-1]
    rows = list(csv.DictReader(open(p, encoding='utf-8')))
    b = next((float(r['mAP']) for r in rows if r['method'].startswith('baseline')), None)
    if b is None:
        missing.append(s)
        continue
    print(f'  seed {s}: nền {100*b:.3f}  <- {p}')
    bases.append(b)
    for r in rows:
        if r['method'] != 'F postproc':
            continue
        k = int(r['param'].split('qe_k=')[1].split(',')[0])
        per_k.setdefault(k, []).append(float(r['mAP']) - b)

if missing:
    print(f'  !! thiếu kết quả cho seed: {missing} — bảng dưới KHÔNG đủ seed')
if not per_k:
    raise SystemExit('Không đọc được improve_eval.csv nào.')
if len(bases) < len(seeds):
    print(f'  !! chỉ tổng hợp được {len(bases)}/{len(seeds)} seed')
print(f'nền: {100*st.fmean(bases):.3f} ± '
      f'{100*st.stdev(bases):.3f} pp  (n={len(bases)} seed)' if len(bases) > 1
      else f'nền: {100*bases[0]:.3f} (n=1)')
print(f'\n{"qe_k":>6} {"Δ trung bình":>14} {"std":>8} {"min":>8} {"max":>8} {"n":>3}')
print('-' * 52)
for k in sorted(per_k):
    v = per_k[k]
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f'{k:>6} {100*st.fmean(v):>+14.3f} {100*sd:>8.3f} '
          f'{100*min(v):>+8.3f} {100*max(v):>+8.3f} {len(v):>3}')
best = max(per_k, key=lambda k: st.fmean(per_k[k]))
v = per_k[best]
sd = st.stdev(v) if len(v) > 1 else 0.0
print(f'\nTốt nhất: qe_k={best} -> {100*st.fmean(v):+.3f} ± {100*sd:.3f} pp')
if len(v) > 1:
    print(f'  Ổn định giữa seed: khoảng [{100*min(v):+.3f}, {100*max(v):+.3f}] pp')
    ok = 100 * min(v) > 0.31          # min(v) là phân số, δ_min tính bằng pp
    print(f'  {"ĐẠT" if ok else "CHƯA ĐẠT"}: mọi seed đều {">" if ok else "<="} δ_min = 0.31 pp')
PY

cat <<'EOF'

################################################################
Đọc kết quả:
  - Mọi seed đều > δ_min -> kết quả vững, đưa vào bài được.
  - qe_k tối ưu giống nhau giữa các seed -> không cần tune, càng tốt.
  - Nếu qe_k tối ưu lệch nhiều -> phải tune, và tune trên tập test là
    gian lận; khi đó chọn một qe_k cố định hợp lý (vd. 20) và báo cáo nó.

Nhắc lại: αQE dùng toàn bộ gallery lúc test. Bảng phải tách dòng
inductive / transductive.
################################################################
EOF
