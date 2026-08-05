#!/usr/bin/env bash
# GIAI ĐOẠN 0 — Đo lường. Không tạo phương pháp mới nào, nhưng quyết định toàn
# bộ phần còn lại của kế hoạch. Không được bỏ qua.
#
#   0.4  verify_training.py        ~10 phút   nền tảng còn sạch không
#   0.1  chẩn đoán frozen-only     ~10 phút   prompt đóng góp bao nhiêu điểm
#   0.2  3 seed baseline           ~11 giờ    thanh sai số δ_min
#   0.3  phân tích epoch           0 phút     best epoch ở đâu, screening có tin được không
#
# Chạy:
#   bash scripts/run_phase0.sh
#   DRY=1 bash scripts/run_phase0.sh     # chỉ in lệnh, không chạy
#   SEEDS="1 2 3 4 5" bash scripts/run_phase0.sh
#   KEEP_CKPT=1 bash scripts/run_phase0.sh   # giữ lại checkpoint (mặc định: xóa)
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"1 2 3"}
EPOCHS=${EPOCHS:-60}
TAG=${TAG:-p0}
KEEP_CKPT=${KEEP_CKPT:-0}

# Base dùng --disable_cross_exchange: cross exchange đã được chứng minh là trung
# tính về hiệu năng nhưng tốn 46.1M tham số và làm chậm mỗi vòng lặp. Dùng nó
# làm base là tự trả giá cho thứ không mua được gì. Nó quay lại ở Giai đoạn 3
# với tư cách một dòng trong bảng ablation.
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
  --lambda_ce=1
  --clip_LN_lr=1e-6
  --prompt_lr=1e-4
  --batch_size=128
  --workers=4
  --test_batch_size=1024
  --disable_augmentation
  --no_resume
)

run() {
  echo; echo "+ python -u -m experiments.hicropl_prompt $*"; echo
  [ "${DRY:-0}" = "1" ] || python -u -m experiments.hicropl_prompt "$@"
}

# Checkpoint 637M params nặng ~2.5GB/run. Giai đoạn 0 KHÔNG cần chúng: 0.3 chỉ
# đọc tb_logs/*/ap_best.npz và runs_summary.csv. Không xóa thì 3 seed = 7.5GB
# nằm lại trong quota home cho tới hết đời.
prune() {   # prune <exp_name>
  [ "$KEEP_CKPT" = "1" ] && { echo "# KEEP_CKPT=1 -> giữ saved_models/$1"; return 0; }
  [ -n "$1" ] || { echo "!! prune: exp_name rỗng, bỏ qua"; return 0; }
  if [ "${DRY:-0}" = "1" ]; then
    echo "+ rm -rf saved_models/$1"
  else
    rm -rf "saved_models/$1"
    echo "# đã xóa saved_models/$1 (dùng KEEP_CKPT=1 để giữ)"
  fi
}

echo "############ 0.4 — Kiểm bất biến của vòng huấn luyện ############"
if [ "${DRY:-0}" = "1" ]; then
  echo "+ python scripts/verify_training.py --all"
else
  python scripts/verify_training.py --all || {
    echo "!! verify_training FAIL — dừng lại. Đừng chạy thí nghiệm trên nền hỏng."; exit 1; }
fi

echo
echo "############ 0.1 — Prompt đóng góp bao nhiêu? ############"
echo "# Eval chỉ với CLIP đóng băng. So con số này với mAP đầy đủ (~78.4)."
echo "#   chênh ~1.5 điểm -> prompt gần như vô dụng, nhảy thẳng tới bỏ residual mix"
echo "#   chênh ~5-8 điểm -> Giai đoạn 1-2 đáng đầu tư đầy đủ"
run "${COMMON[@]}" \
  --exp_name="${TAG}_frozen_only" --seed=1 --epochs=1 \
  --eval_frozen_only --save_top_k=0

echo
echo "############ 0.2 — Thanh sai số: ${SEEDS} ############"
for s in $SEEDS; do
  run "${COMMON[@]}" \
    --exp_name="${TAG}_base_s${s}" --seed="$s" --epochs="$EPOCHS" --save_top_k=1
  prune "${TAG}_base_s${s}"
done

echo
echo "############ 0.3 — Phân tích ############"
if [ "${DRY:-0}" = "1" ]; then
  echo "+ python scripts/summarize_runs.py --filter ${TAG}_base --epochs"
  echo "+ python scripts/paired_test.py --a ... --b ..."
else
  python scripts/summarize_runs.py --filter "${TAG}_base" --epochs

  echo
  echo "=== δ_min: kiểm cặp giữa MỌI cặp seed ==="
  echo "Hai seed cùng cấu hình nên khác biệt thật bằng 0. Nửa độ rộng KTC lớn"
  echo "nhất qua các cặp chính là δ_min — ngưỡng mà dưới đó bạn không kết luận được."
  echo "Dùng --from-summary thay vì glob: run bị ngắt giữa chừng vẫn kịp ghi"
  echo "ap_best.npz nhưng không có dòng summary, glob sẽ nhặt nhầm chúng."
  for a in $SEEDS; do
    for b in $SEEDS; do
      [ "$a" -lt "$b" ] || continue
      echo; echo "--- seed$a vs seed$b ---"
      python scripts/paired_test.py \
        --from-summary "${SUMMARY:-runs_summary.csv}" \
        --a "${TAG}_base_s${a}" --b "${TAG}_base_s${b}" \
        --name-a "seed$a" --name-b "seed$b" || true
    done
  done
fi

cat <<'EOF'

################################################################
GIAI ĐOẠN 0 XONG. Ba con số cần ghi lại trước khi đi tiếp:

  1. mAP frozen-only  = ____   (so với ~78.4 đầy đủ)
  2. δ_min            = ____ pp  (từ kiểm cặp cùng-cấu-hình)
  3. best epoch median= ____   (nếu << 60 thì rút ngắn EPOCHS)

Rẽ nhánh:
  - Nếu (1) chỉ kém 1-2 điểm  -> bỏ Giai đoạn 1, sang thẳng residual mix
  - Nếu Spearman screening ≥ 0.9 -> Giai đoạn 1 dùng EPOCHS=20, tiết kiệm 2/3
  - Mọi chênh lệch < (2) về sau đều phải báo cáo là "không phân biệt được"
################################################################
EOF
