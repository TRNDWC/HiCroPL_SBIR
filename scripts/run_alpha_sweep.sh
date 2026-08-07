#!/usr/bin/env bash
# Quét α: train MỘT run có giữ checkpoint, rồi quét α tại thời điểm eval.
#
# Cơ sở (docs/findings-and-directions.md §3.6, §7.1, §7.3):
#   Giai đoạn 3a đo được α ≥ 0.5 vì α học được luôn tăng về 1, và mọi mức tăng
#   đều làm mAP tệ đi đơn điệu. Nửa còn lại của đường cong — α < 0.5 — hoàn toàn
#   chưa biết. Và α = 0 chính là frozen-only, con số còn thiếu từ Giai đoạn 0.
#
# Chi phí: 1 lần train (~40 phút) + 1 lần trích đặc trưng. Encoder chỉ chạy một
# lần cho toàn dải α vì hai nhánh không phụ thuộc α.
#
# LƯU Ý: Giai đoạn 2 và 3 chạy với --save_top_k=0 nên KHÔNG có checkpoint nào
# dùng lại được. Bắt buộc train lại một run ở đây.
#
# Chạy:
#   bash scripts/run_alpha_sweep.sh
#   SEED=2 bash scripts/run_alpha_sweep.sh
#   ALPHAS="0 0.25 0.5 0.75 1.0" bash scripts/run_alpha_sweep.sh
#   DRY=1 bash scripts/run_alpha_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=${SEED:-1}
EPOCHS=${EPOCHS:-8}
TAG=${TAG:-p4}
ALPHAS=${ALPHAS:-"0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.85 1.0"}
EXP="${TAG}_alphasweep_s${SEED}"

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
)

echo "############ 1/2 — Train một run, GIỮ checkpoint ############"
echo "# --save_top_k=1: checkpoint ~2.5GB, cần cho bước quét. Xoá sau khi xong."
if [ "${DRY:-0}" = "1" ]; then
  echo "+ python -u -m experiments.hicropl_prompt ${COMMON[*]} --exp_name=$EXP --seed=$SEED --epochs=$EPOCHS --save_top_k=1"
else
  python -u -m experiments.hicropl_prompt "${COMMON[@]}" \
    --exp_name="$EXP" --seed="$SEED" --epochs="$EPOCHS" --save_top_k=1
fi

# run_dir mới nhất của experiment vừa chạy.
# KHÔNG dùng `ls | sort | tail -1`: TensorBoardLogger tạo tb_logs/<exp>/version_N/
# nằm cạnh thư mục run, và 'v' (0x76) > '2' (0x32) nên version_0 luôn thắng khi
# sắp xếp. Lọc theo config.json — chỉ thư mục run mới có file đó.
RUN_DIR=$(find "tb_logs/$EXP" -mindepth 2 -maxdepth 2 -name config.json 2>/dev/null \
          | xargs -r -n1 dirname | sort | tail -1 || true)
if [ "${DRY:-0}" = "1" ]; then
  echo; echo "+ python scripts/sweep_alpha.py --run_dir <run_dir> --alphas $ALPHAS --save_ap"
  exit 0
fi
[ -n "$RUN_DIR" ] || { echo "!! không tìm thấy tb_logs/$EXP/*/ — train có chạy không?"; exit 1; }

echo
echo "############ 2/2 — Quét α trên checkpoint ############"
python scripts/sweep_alpha.py --run_dir "${RUN_DIR%/}" --alphas $ALPHAS --save_ap

cat <<EOF

################################################################
Checkpoint còn ở saved_models/$EXP (~2.5GB). Xoá khi không cần:
    rm -rf saved_models/$EXP

Nếu muốn kết luận chặt về chênh lệch nhỏ hơn δ_min = 0.31 pp,
--save_ap đã ghi vector AP cho từng α:

    python scripts/paired_test.py \\
      ${RUN_DIR%/}/ap_alpha0.50.npz ${RUN_DIR%/}/ap_alpha0.30.npz

Lưu ý diễn giải: đây là quét α tại EVAL trên một model đã train ở
α=0.5. Nếu đỉnh lệch khỏi 0.5, bước tiếp theo là train LẠI ở α đó
để biết tối ưu chung của (train, eval) — hai thứ không nhất thiết
trùng nhau.
################################################################
EOF
