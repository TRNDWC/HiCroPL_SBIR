#!/usr/bin/env bash
# GIAI ĐOẠN 5b — so công bằng giữa 'rel' và 'direct'.
#
# Vì sao cần: ở phase 5, cả ba chế độ dùng chung lambda_visual_cross=0.1, nhưng
# độ lớn loss khác hẳn nhau:
#     legacy 0.0373   direct 0.1322   rel 0.0088
# nên 'rel' đang được áp với trọng số hiệu dụng NHỎ HƠN 'direct' 15 lần. Kết
# quả "rel kém hơn direct 0.053 pp" vì vậy KHÔNG kết luận được — đây là lỗi
# thiết kế thí nghiệm, không phải phát hiện.
#
# lambda khớp = 0.1 x (0.1322/0.0088) = 1.50. Quét bao quanh vì độ lớn loss
# thay đổi trong lúc train nên khớp tại một điểm chỉ là xấp xỉ.
#
# Nếu 'rel' ở trọng số đúng VƯỢT 'direct' thì lập luận modality gap được xác
# nhận; nếu vẫn kém thì lập luận đó bị bác bỏ ở mức trọng số này.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"1 2 3"}
EPOCHS=${EPOCHS:-8}
TAG=${TAG:-p5b}
LAMBDAS=${LAMBDAS:-"0.5 1.5 4.0"}

COMMON=(
  --dataset=sketchy_ext --data_dir=../data/Sketchy
  --gpt_text_file=gpt_file/sketchy_ext.json
  --n_ctx=2 --prompt_depth=12 --language_depth=1 --cross_layer=6
  --disable_cross_exchange --lambda_cross_modal=1.0 --lambda_ce=1.0
  --weight_decay=0 --clip_LN_lr=1e-6 --prompt_lr=1e-4
  --batch_size=128 --workers=4 --test_batch_size=1024
  --disable_augmentation --no_resume --save_top_k=0
  --enhance_text --text_align_mode=rel
)

for lv in $LAMBDAS; do
  for s in $SEEDS; do
    name="${TAG}_rel_lv${lv}_s${s}"
    echo; echo "+ rel, lambda_visual_cross=$lv, seed=$s"
    if [ "${DRY:-0}" = "1" ]; then
      echo "  python -u -m experiments.hicropl_prompt ... --exp_name=$name --lambda_visual_cross=$lv"
    else
      python -u -m experiments.hicropl_prompt "${COMMON[@]}" \
        --exp_name="$name" --seed="$s" --epochs="$EPOCHS" --lambda_visual_cross="$lv"
      rm -rf "saved_models/$name"
    fi
  done
done

[ "${DRY:-0}" = "1" ] && exit 0
echo; echo "############ Kết quả ############"
python scripts/summarize_runs.py --filter "${TAG}_rel" --group-by cfg_lambda_visual_cross

cat <<'EOF'

################################################################
So với (cùng 3 seed, xem docs §8c):
  base                        78.368
  text_direct (lv=0.1)        78.489   +0.121 ± 0.023
  text_rel    (lv=0.1)        78.436   +0.068 ± 0.015

Dùng so CẶP theo seed, không so trung bình: các biến thể dùng
chung seed nên δ_min=0.31 (ngưỡng KHÔNG cặp) quá bảo thủ ở đây.
################################################################
EOF
