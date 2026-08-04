#!/usr/bin/env bash
# GIAI ĐOẠN 1 — Quét capacity (n_ctx).
#
# Vì sao trước mọi thứ khác: n_ctx là nút capacity RẺ NHẤT và đang ở mức tối
# thiểu. Tham số của mapper tỉ lệ d_model^2 chứ không phải độ dài chuỗi, nên
# n_ctx 2->16 chỉ thêm ~0.6% tham số. Hiện prompt chiếm 3.8% chuỗi ViT (2 token
# trên 52). Nếu mô hình đang thiếu capacity một cách giả tạo thì MỌI so sánh
# phương pháp sau đó đều diễn ra ở điểm vận hành sai.
#
# Hai bước: sàng lọc 1 seed -> xác nhận 3 seed cho 2 giá trị tốt nhất.
#
# Chạy:
#   bash scripts/run_phase1.sh                    # sàng lọc
#   BEST="8 16" bash scripts/run_phase1.sh        # xác nhận 3 seed
#   DRY=1 bash scripts/run_phase1.sh
#
# EPOCHS: nếu Giai đoạn 0.3 báo Spearman >= 0.9 thì đặt EPOCHS=20 cho bước sàng
# lọc, tiết kiệm 2/3 thời gian. Nếu không, giữ nguyên 60.
set -euo pipefail
cd "$(dirname "$0")/.."

NCTX_LIST=${NCTX_LIST:-"2 4 8 16"}
SCREEN_EPOCHS=${SCREEN_EPOCHS:-60}
CONFIRM_EPOCHS=${CONFIRM_EPOCHS:-60}
SEEDS=${SEEDS:-"1 2 3"}
BEST=${BEST:-""}
TAG=${TAG:-p1}

COMMON=(
  --dataset=sketchy_ext
  --data_dir=../data/Sketchy
  --gpt_text_file=gpt_file/sketchy_ext.json
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

if [ -z "$BEST" ]; then
  echo "############ 1a — Sàng lọc n_ctx: ${NCTX_LIST} (1 seed, ${SCREEN_EPOCHS} epoch) ############"
  echo "# save_top_k=0: checkpoint 637M params nặng ~2.5GB/run, sàng lọc không cần giữ."
  for n in $NCTX_LIST; do
    run "${COMMON[@]}" \
      --exp_name="${TAG}_nctx${n}_s1" --seed=1 --n_ctx="$n" \
      --epochs="$SCREEN_EPOCHS" --save_top_k=0
  done

  if [ "${DRY:-0}" != "1" ]; then
    echo; echo "############ Kết quả sàng lọc ############"
    python scripts/summarize_runs.py --filter "${TAG}_nctx" --group-by cfg_n_ctx
  fi

  cat <<EOF

################################################################
Sàng lọc xong. Chọn 2 giá trị n_ctx tốt nhất rồi chạy:

    BEST="8 16" bash scripts/run_phase1.sh

Lưu ý khi đọc bảng: chênh lệch nhỏ hơn δ_min (từ Giai đoạn 0) là
KHÔNG kết luận được. Nếu cả 4 giá trị nằm trong δ_min của nhau thì
n_ctx không phải nút thắt — giữ n_ctx=2 và sang Giai đoạn 2.
################################################################
EOF
  exit 0
fi

echo "############ 1b — Xác nhận n_ctx ∈ {${BEST}} × seed {${SEEDS}} ############"
for n in $BEST; do
  for s in $SEEDS; do
    run "${COMMON[@]}" \
      --exp_name="${TAG}_nctx${n}_s${s}" --seed="$s" --n_ctx="$n" \
      --epochs="$CONFIRM_EPOCHS" --save_top_k=1
  done
done

if [ "${DRY:-0}" != "1" ]; then
  echo; echo "############ Bảng ablation n_ctx ############"
  python scripts/summarize_runs.py --filter "${TAG}_nctx" --group-by cfg_n_ctx --epochs

  set -- $BEST
  if [ $# -ge 2 ]; then
    echo; echo "=== Kiểm cặp giữa hai giá trị n_ctx tốt nhất ==="
    python scripts/paired_test.py \
      --a "tb_logs/${TAG}_nctx$1_s*/*/ap_best.npz" \
      --b "tb_logs/${TAG}_nctx$2_s*/*/ap_best.npz" \
      --name-a "n_ctx=$1" --name-b "n_ctx=$2" || true
  fi
fi

cat <<'EOF'

################################################################
GIAI ĐOẠN 1 XONG. Chốt một giá trị n_ctx và dùng CỐ ĐỊNH cho mọi
thí nghiệm về sau — nếu không, các so sánh phương pháp ở Giai đoạn
2-3 sẽ lẫn ảnh hưởng của capacity vào.
################################################################
EOF
