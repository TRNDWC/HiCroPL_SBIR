#!/usr/bin/env bash
# GIAI ĐOẠN 3a — Trọng số trộn residual học được.
#
# Bằng chứng dẫn tới đây, cả ba đều từ Giai đoạn 2:
#   - lambda_ce=0  -> best TỆ HƠN 0.83 pp, suy giảm không đổi   (L4 không phải thủ phạm)
#   - weight_decay -> đổi best 0.014 pp, dưới δ_min 20 lần       (regularization chung trơ)
#   - n_ctx 2->16  -> suy giảm 0.36 -> 3.54 pp, best epoch 3->0  (capacity làm tệ hơn)
#
# Bỏ loss thì tệ hơn, thêm regularization thì trơ, thêm capacity thì tệ hơn.
# Nên suy giảm không đến từ một khuyết tật cụ thể nào mà là hệ quả nội tại của
# việc khớp phân phối huấn luyện. Cần một ràng buộc CÓ MỤC TIÊU.
#
# Điểm tấn công: residual mix hiện là thứ DUY NHẤT giữ tính tổng quát, và nó
# làm việc đó một cách thô bạo — ép cứng tỉ lệ 1:1, không học được, giống hệt
# nhau cho cả hai modality. Nhưng CLIP mạnh trên ảnh và yếu trên sketch, nên
# nhánh sketch đang bị neo vào chính phần đặc trưng kém chất lượng nhất.
#
#   feat = norm( a * prompted + (1-a) * frozen ),  a = sigmoid(theta)
#
# theta=0 -> a=0.5 -> trùng ĐÚNG hành vi cũ. Hai tham số cho cả mô hình.
#
# Đọc gì:
#   alpha_photo / alpha_sketch có trong metrics_epoch.csv theo từng epoch.
#   - alpha_sketch > alpha_photo  -> bất đối xứng có thật, một phát hiện gọn
#   - alpha -> 0                  -> nhánh prompted đang GÂY HẠI, tự nó là kết quả mạnh
#   - alpha đứng yên 0.5          -> gradient không tới được, kiểm mix_alpha_lr
#
# Chạy:
#   bash scripts/run_phase3.sh                 # 3 seed base + 3 seed alpha, ~4h
#   MIX_LRS="1e-3 1e-2" bash scripts/run_phase3.sh   # quét lr của alpha trước
#   DRY=1 bash scripts/run_phase3.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-8}
SEEDS=${SEEDS:-"1 2 3"}
MIX_LRS=${MIX_LRS:-"1e-3"}
TAG=${TAG:-p3}
KEEP_CKPT=${KEEP_CKPT:-0}
SUMMARY=${SUMMARY:-runs_summary.csv}

COMMON=(
  --dataset=sketchy_ext
  --data_dir=../data/Sketchy
  --gpt_text_file=gpt_file/sketchy_ext.json
  --n_ctx=2
  --prompt_depth=12
  --language_depth=1
  --cross_layer=6
  --disable_cross_exchange
  --lambda_cross_modal=1.0
  --lambda_ce=1.0
  --weight_decay=0
  --clip_LN_lr=1e-6
  --prompt_lr=1e-4
  --batch_size=128
  --workers=4
  --test_batch_size=1024
  --disable_augmentation
  --no_resume
  --save_top_k=0
)

run() {
  echo; echo "+ python -u -m experiments.hicropl_prompt $*"; echo
  [ "${DRY:-0}" = "1" ] || python -u -m experiments.hicropl_prompt "$@"
}

prune() {
  [ "$KEEP_CKPT" = "1" ] && return 0
  [ -n "${1:-}" ] || return 0
  [ "${DRY:-0}" = "1" ] && { echo "+ rm -rf saved_models/$1"; return 0; }
  rm -rf "saved_models/$1"
}

echo "############ 3a — Baseline đối chứng ($SEEDS) ############"
echo "# Cùng cấu hình, KHÔNG bật alpha. Cần chạy lại dưới TAG này để phép kiểm"
echo "# cặp có hai nhóm cùng số seed và cùng thứ tự query."
for s in $SEEDS; do
  name="${TAG}_base_s${s}"
  run "${COMMON[@]}" --exp_name="$name" --seed="$s" --epochs="$EPOCHS"
  prune "$name"
done

for mlr in $MIX_LRS; do
  echo
  echo "############ 3a — learn_mix_alpha, mix_alpha_lr=${mlr} ############"
  for s in $SEEDS; do
    name="${TAG}_alpha${mlr}_s${s}"
    run "${COMMON[@]}" --exp_name="$name" --seed="$s" --epochs="$EPOCHS" \
        --learn_mix_alpha --mix_alpha_lr="$mlr"
    prune "$name"
  done
done

if [ "${DRY:-0}" != "1" ]; then
  echo; echo "############ Kết quả ############"
  python scripts/summarize_runs.py --summary "$SUMMARY" --filter "${TAG}_" \
    --group-by cfg_learn_mix_alpha cfg_mix_alpha_lr --degradation

  for mlr in $MIX_LRS; do
    echo; echo "=== Kiểm cặp: base vs alpha(lr=$mlr) ==="
    python scripts/paired_test.py --from-summary "$SUMMARY" \
      --a "${TAG}_base_s" --b "${TAG}_alpha${mlr}_s" \
      --name-a base --name-b "alpha_lr${mlr}" || true
  done

  echo; echo "=== Quỹ đạo alpha theo epoch ==="
  for f in tb_logs/${TAG}_alpha*/*/metrics_epoch.csv; do
    [ -e "$f" ] || continue
    echo "--- $f"
    python - "$f" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8')))
cols = [c for c in ('epoch', 'alpha_photo', 'alpha_sketch', 'mAP') if c in rows[0]]
if 'alpha_photo' not in cols:
    print('  (không có cột alpha — kiểm lại --learn_mix_alpha)')
else:
    print('  ' + '  '.join(f'{c:>12}' for c in cols))
    for r in rows:
        print('  ' + '  '.join(f'{r[c]:>12}' for c in cols))
PY
  done
fi

cat <<'EOF'

################################################################
Cách đọc, theo thứ tự quan trọng:

1. QUỸ ĐẠO ALPHA quan trọng hơn mAP. Nó là lời khai của chính mô
   hình về việc nên tin nhánh prompted bao nhiêu — thứ mà cho tới
   giờ ta chỉ suy đoán gián tiếp.

2. alpha_sketch != alpha_photo -> bất đối xứng có thật giữa hai
   modality. Đủ để thành một hình trong bài báo.

3. alpha giảm dần về 0 -> nhánh prompted gây hại, và residual mix
   1:1 cố định đang che giấu điều đó. Đây là kết quả MẠNH, dù mAP
   không đổi: nó nói cơ chế prompt hiện tại không đóng góp.

4. Chỉ khi (3) rõ mới đáng đầu tư vào 3b (thay residual mix bằng
   relational distillation). Nếu alpha ở lại quanh 0.5 thì nhánh
   prompted đang đóng góp thật, và 3b sẽ rủi ro hơn nhiều.

Nhắc lại δ_min ≈ 0.31 pp. Chênh lệch mAP nhỏ hơn mức đó là không
kết luận được — nhưng quỹ đạo alpha thì đọc trực tiếp, không cần
kiểm định.
################################################################
EOF
