#!/usr/bin/env bash
# GIAI ĐOẠN 2 — Truy nguồn overfit.
#
# Bằng chứng từ Giai đoạn 0: mô hình đạt đỉnh ở epoch 2-3 rồi TỤT 1.14 pp trong
# khi train loss vẫn giảm. Nút thắt không phải capacity mà là tổng quát hoá.
# Điều này đảo ngược ưu tiên ban đầu (tăng n_ctx), nên n_ctx bị đẩy xuống cuối
# và chạy với kỳ vọng ngược: capacity lớn hơn có thể làm overfit NHANH hơn.
#
# Chỉ số đo chính là SUY GIẢM (best − last), không phải mAP tuyệt đối. Nó đo
# trong một run nên khử phần lớn dao động seed: std ~0.12 pp so với 0.31 pp của
# best-mAP. Nhờ vậy 1 seed đã đủ để sàng lọc.
#
# Chạy:
#   bash scripts/run_phase2.sh              # 2a: truy nguồn (6 run, ~4h)
#   STAGE=2 CONFIRM="lce0 wd1e-2" bash scripts/run_phase2.sh   # xác nhận 3 seed
#   STAGE=3 bash scripts/run_phase2.sh      # 2c: n_ctx (4 run, ~2.7h)
#   DRY=1 bash scripts/run_phase2.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# best epoch = 3 -> 8 epoch là đủ: 3 để lên đỉnh, 5 để thấy hình dạng suy giảm.
# Giai đoạn 0 chạy 10 epoch mất ~50 phút, nên mỗi run ở đây ~40 phút.
EPOCHS=${EPOCHS:-8}
SEEDS=${SEEDS:-"1 2 3"}
STAGE=${STAGE:-1}
TAG=${TAG:-p2}
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

# ---------------------------------------------------------------- 2a
if [ "$STAGE" = "1" ]; then
  echo "############ 2a — Truy nguồn overfit (1 seed, ${EPOCHS} epoch) ############"
  echo "# Giả thuyết 1: L4 (cross-entropy trên 104 lớp ĐÃ THẤY) là nguồn overfit."
  echo "#   Nó ép đặc trưng phân biệt lớp train, trong khi test toàn lớp chưa thấy."
  echo "#   LƯU Ý confound: lambda_ce=0 cũng làm prompt văn bản ngừng học hẳn"
  echo "#   (gradient của chúng chỉ đến từ L4). Nếu lambda_ce=0 thắng, cần một"
  echo "#   thí nghiệm tiếp để tách hai hiệu ứng."
  for lce in 1.0 0.5 0.0; do
    name="${TAG}_lce${lce}_s1"
    run "${COMMON[@]}" --exp_name="$name" --seed=1 --epochs="$EPOCHS" \
        --lambda_ce="$lce" --weight_decay=0
    prune "$name"
  done

  echo
  echo "# Giả thuyết 2: thiếu regularization. weight_decay hiện đang là 0."
  for wd in 1e-4 1e-2; do
    name="${TAG}_wd${wd}_s1"
    run "${COMMON[@]}" --exp_name="$name" --seed=1 --epochs="$EPOCHS" \
        --lambda_ce=1.0 --weight_decay="$wd"
    prune "$name"
  done

  if [ "${DRY:-0}" != "1" ]; then
    echo; echo "############ Kết quả 2a ############"
    python scripts/summarize_runs.py --summary "$SUMMARY" --filter "${TAG}_" \
      --group-by cfg_lambda_ce cfg_weight_decay --degradation
  fi

  cat <<EOF

################################################################
Đọc cột SUY GIẢM, không đọc cột best.

  - Suy giảm giảm mạnh khi lambda_ce nhỏ  -> L4 là nguồn overfit
  - Suy giảm giảm mạnh khi weight_decay lớn -> thiếu regularization
  - Cả hai đều không đổi -> nguồn nằm chỗ khác (nghi tiếp: L1 InfoNCE
    trên lớp đã thấy, hoặc chính LayerNorm đang trôi)

Mốc so sánh từ Giai đoạn 0: suy giảm 1.14 pp, best 78.37.
Cấu hình nào giữ được best mà suy giảm < 0.5 pp là ứng viên.

Rồi xác nhận 3 seed:
    STAGE=2 CONFIRM="lce0.0 wd1e-2" bash scripts/run_phase2.sh
################################################################
EOF
  exit 0
fi

# ---------------------------------------------------------------- 2b
if [ "$STAGE" = "2" ]; then
  : "${CONFIRM:?Cần CONFIRM=\"lce0.0 wd1e-2\" — hậu tố của các cấu hình muốn xác nhận}"
  echo "############ 2b — Xác nhận {${CONFIRM}} × seed {${SEEDS}} ############"
  for cfg in $CONFIRM; do
    case "$cfg" in
      lce*) extra=(--lambda_ce="${cfg#lce}" --weight_decay=0) ;;
      wd*)  extra=(--lambda_ce=1.0 --weight_decay="${cfg#wd}") ;;
      base) extra=(--lambda_ce=1.0 --weight_decay=0) ;;
      *) echo "!! không hiểu cấu hình '$cfg' (dùng lce<val>, wd<val>, hoặc base)"; exit 1 ;;
    esac
    for s in $SEEDS; do
      name="${TAG}c_${cfg}_s${s}"
      run "${COMMON[@]}" --exp_name="$name" --seed="$s" --epochs="$EPOCHS" "${extra[@]}"
      prune "$name"
    done
  done

  if [ "${DRY:-0}" != "1" ]; then
    echo; echo "############ Bảng xác nhận ############"
    python scripts/summarize_runs.py --summary "$SUMMARY" --filter "${TAG}c_" \
      --group-by cfg_lambda_ce cfg_weight_decay --degradation

    set -- $CONFIRM
    if [ $# -ge 2 ]; then
      echo; echo "=== Kiểm cặp: $1 vs $2 ==="
      python scripts/paired_test.py --from-summary "$SUMMARY" \
        --a "${TAG}c_$1_s" --b "${TAG}c_$2_s" --name-a "$1" --name-b "$2" || true
    fi
  fi

  cat <<'EOF'

################################################################
Nhắc lại δ_min ≈ 0.31 pp từ Giai đoạn 0: chênh lệch best-mAP nhỏ hơn
mức này là KHÔNG kết luận được. Cột SUY GIẢM thì nhạy hơn — dùng nó.
################################################################
EOF
  exit 0
fi

# ---------------------------------------------------------------- 2c
if [ "$STAGE" = "3" ]; then
  echo "############ 2c — n_ctx với KỲ VỌNG ĐẢO NGƯỢC (1 seed) ############"
  echo "# Ban đầu tôi xếp tăng n_ctx là ưu tiên số một, với lập luận mô hình"
  echo "# thiếu capacity. Giai đoạn 0 bác bỏ giả định đó: mô hình overfit sau"
  echo "# 3 epoch. Vậy nên thí nghiệm này giờ kiểm điều ngược lại — capacity"
  echo "# lớn hơn có làm overfit NHANH hơn không. Đọc cột SUY GIẢM là chính."
  for n in 2 4 8 16; do
    name="${TAG}_nctx${n}_s1"
    run "${COMMON[@]/--n_ctx=2/--n_ctx=$n}" \
        --exp_name="$name" --seed=1 --epochs="$EPOCHS" --lambda_ce=1.0 --weight_decay=0
    prune "$name"
  done

  if [ "${DRY:-0}" != "1" ]; then
    echo; echo "############ Kết quả n_ctx ############"
    python scripts/summarize_runs.py --summary "$SUMMARY" --filter "${TAG}_nctx" \
      --group-by cfg_n_ctx --degradation
  fi

  cat <<'EOF'

################################################################
Nếu SUY GIẢM tăng theo n_ctx -> xác nhận đây là bài toán tổng quát
hoá chứ không phải capacity, và củng cố hướng đi vào regularization
thay vì mở rộng cơ chế prompt.
################################################################
EOF
  exit 0
fi

echo "STAGE không hợp lệ: $STAGE (dùng 1, 2 hoặc 3)"; exit 1
