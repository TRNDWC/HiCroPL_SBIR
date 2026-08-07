#!/usr/bin/env bash
# GIAI ĐOẠN 5 — hai hướng cuối chưa từng thử: D (sửa công thức text) và E (SupCon).
#
# D — §5.D: `1 - cos(a+b, a)` chỉ là reparameterization đơn điệu của cosine với
#     gradient bão hoà. Nghiêm trọng hơn, `loss_cons_visual_cross` kéo đặc trưng
#     ẢNH về đặc trưng TEXT, đi ngược modality gap của CLIP (cos(image,text)
#     ~0.2-0.3 ngay cả với cặp khớp hoàn hảo). Chế độ 'rel' thay bằng ràng buộc
#     THỨ HẠNG lớp.
# E — §5.E: NT-Xent coi ảnh CÙNG LỚP là negative. Batch 128 / 104 lớp cho ~2.4
#     false negative mỗi hàng, và chúng là những negative giống nhất.
#
# Cả hai opt-in, mặc định tắt, nên không đụng kết quả đã có.
# Nhắc: mọi chênh lệch < δ_min = 0.31 pp là không kết luận được.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"1 2 3"}
EPOCHS=${EPOCHS:-8}
TAG=${TAG:-p5}

COMMON=(
  --dataset=sketchy_ext --data_dir=../data/Sketchy
  --gpt_text_file=gpt_file/sketchy_ext.json
  --n_ctx=2 --prompt_depth=12 --language_depth=1 --cross_layer=6
  --disable_cross_exchange --lambda_cross_modal=1.0 --lambda_ce=1.0
  --weight_decay=0 --clip_LN_lr=1e-6 --prompt_lr=1e-4
  --batch_size=128 --workers=4 --test_batch_size=1024
  --disable_augmentation --no_resume --save_top_k=0
)

run() {
  echo; echo "+ python -u -m experiments.hicropl_prompt $*"; echo
  [ "${DRY:-0}" = "1" ] || python -u -m experiments.hicropl_prompt "$@"
}

# Sàng lọc 1 seed trước; chỉ cấu hình nào vượt δ_min mới xác nhận 3 seed.
declare -a VARIANTS=(
  "base:"
  "supcon:--supcon"
  "text_legacy:--enhance_text --text_align_mode=legacy"
  "text_direct:--enhance_text --text_align_mode=direct"
  "text_rel:--enhance_text --text_align_mode=rel"
  "supcon_text_rel:--supcon --enhance_text --text_align_mode=rel"
)

for v in "${VARIANTS[@]}"; do
  name="${v%%:*}"; flags="${v#*:}"
  for s in $SEEDS; do
    run "${COMMON[@]}" --exp_name="${TAG}_${name}_s${s}" --seed="$s" \
        --epochs="$EPOCHS" $flags
    [ "${DRY:-0}" = "1" ] || rm -rf "saved_models/${TAG}_${name}_s${s}"
  done
  [ "${SCREEN_ONLY:-0}" = "1" ] && break
done

[ "${DRY:-0}" = "1" ] && exit 0
echo; echo "############ Kết quả ############"
python scripts/summarize_runs.py --filter "${TAG}_" \
  --group-by cfg_supcon cfg_enhance_text cfg_text_align_mode --degradation

cat <<'EOF'

################################################################
Nền để so: 78.368 ± 0.322 (3 seed, xem docs §8b).
Chênh lệch < δ_min = 0.31 pp là KHÔNG kết luận được — dùng
scripts/paired_test.py nếu cần phân giải chặt hơn.

Nhắc: 'text_legacy' là bản gốc, đưa vào để tách riêng ảnh hưởng
của việc BẬT enhance_text với ảnh hưởng của việc ĐỔI công thức.
################################################################
EOF
