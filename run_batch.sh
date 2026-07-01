#!/bin/bash

# Dừng script ngay lập tức nếu có bất kỳ lệnh nào bị lỗi
set -e

# Khởi tạo môi trường
source activate hicropl
export PYTHONNOUSERSITE=True
cd /home/hoangu/HiCroPL_SBIR || exit

echo "======================================================"
echo "BẮT ĐẦU CHẠY CHUỖI THÍ NGHIỆM (ABLATION STUDY)"
echo "======================================================"

# =================================================================
# THÍ NGHIỆM 1: BASELINE THUẦN TÚY
# (Không Cross-Exchange, Không Augmentation, Không Enhance Text)
# =================================================================
EXP1_NAME="clipat_hicropl_1_baseline"
echo ">>> [1/5] Đang chạy thí nghiệm: $EXP1_NAME ..."
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.hicropl_prompt \
  --exp_name=$EXP1_NAME \
  --dataset=sketchy_ext \
  --data_dir=../data/Sketchy_kag \
  --n_ctx=2 \
  --prompt_depth=12 \
  --language_depth=1 \
  --cross_layer=6 \
  --lambda_cross_modal=1.0 \
  --lambda_consistency=1.0 \
  --lambda_ce=1.0 \
  --clip_LN_lr=1e-6 \
  --prompt_lr=1e-4 \
  --batch_size=128 \
  --workers=4 \
  --test_batch_size=1024 \
  --epochs=6 \
  --disable_cross_exchange \
  --disable_augmentation

echo ">>> Thí nghiệm $EXP1_NAME ĐÃ XONG!"
rm -rf "saved_models/"
echo "------------------------------------------------------"


# =================================================================
# THÍ NGHIỆM 2: + CROSS-EXCHANGE
# (Bật Cross-Exchange, Không Augmentation, Không Enhance Text)
# =================================================================
EXP2_NAME="clipat_hicropl_2_add_crossexchange"
echo ">>> [2/5] Đang chạy thí nghiệm: $EXP2_NAME ..."
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.hicropl_prompt \
  --exp_name=$EXP2_NAME \
  --dataset=sketchy_ext \
  --data_dir=../data/Sketchy_kag \
  --n_ctx=2 \
  --prompt_depth=12 \
  --language_depth=1 \
  --cross_layer=6 \
  --lambda_cross_modal=1.0 \
  --lambda_consistency=1.0 \
  --lambda_ce=1.0 \
  --clip_LN_lr=1e-6 \
  --prompt_lr=1e-4 \
  --batch_size=128 \
  --workers=4 \
  --test_batch_size=1024 \
  --epochs=6 \
  --disable_augmentation

echo ">>> Thí nghiệm $EXP2_NAME ĐÃ XONG!"
rm -rf "saved_models/"
echo "------------------------------------------------------"


# =================================================================
# THÍ NGHIỆM 3: + AUGMENTATION (Tức là Full HiCroPL gốc)
# (Bật Cross-Exchange, Bật Augmentation, Không Enhance Text)
# =================================================================
EXP3_NAME="clipat_hicropl_3_add_augmentation"
echo ">>> [3/5] Đang chạy thí nghiệm: $EXP3_NAME ..."
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.hicropl_prompt \
  --exp_name=$EXP3_NAME \
  --dataset=sketchy_ext \
  --data_dir=../data/Sketchy_kag \
  --n_ctx=2 \
  --prompt_depth=12 \
  --language_depth=1 \
  --cross_layer=6 \
  --lambda_cross_modal=1.0 \
  --lambda_consistency=1.0 \
  --lambda_ce=1.0 \
  --clip_LN_lr=1e-6 \
  --prompt_lr=1e-4 \
  --batch_size=128 \
  --workers=4 \
  --test_batch_size=1024 \
  --epochs=10

echo ">>> Thí nghiệm $EXP3_NAME ĐÃ XONG!"
rm -rf "saved_models/"
echo "------------------------------------------------------"


# =================================================================
# THÍ NGHIỆM 4: + ENHANCE TEXT (LR Config 1)
# (Full + Dynamic Prompting Forward Mix, LR: LN=1e-5, Prompt=1e-5)
# =================================================================
EXP4_NAME="clipat_hicropl_4_add_enhance_lr_1e5"
echo ">>> [4/5] Đang chạy thí nghiệm: $EXP4_NAME ..."
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.hicropl_prompt \
  --exp_name=$EXP4_NAME \
  --dataset=sketchy_ext \
  --data_dir=../data/Sketchy_kag \
  --gpt_text_file='gpt_file/sketchy_ext.json' \
  --n_ctx=2 \
  --prompt_depth=12 \
  --language_depth=1 \
  --cross_layer=6 \
  --lambda_cross_modal=1.0 \
  --lambda_consistency=1.0 \
  --lambda_ce=1.0 \
  --clip_LN_lr=1e-5 \
  --prompt_lr=1e-5 \
  --batch_size=128 \
  --workers=4 \
  --test_batch_size=1024 \
  --epochs=6 \
  --enhance_text \
  --lambda_text_consistency=2.0 \
  --lambda_visual_cross=0.2

echo ">>> Thí nghiệm $EXP4_NAME ĐÃ XONG!"
rm -rf "saved_models/"
echo "------------------------------------------------------"


# =================================================================
# THÍ NGHIỆM 5: + ENHANCE TEXT (LR Config 2)
# (Full + Dynamic Prompting Forward Mix, LR: LN=1e-6, Prompt=1e-4)
# =================================================================
EXP5_NAME="clipat_hicropl_5_add_enhance_lr_1e6_1e4"
echo ">>> [5/5] Đang chạy thí nghiệm: $EXP5_NAME ..."
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.hicropl_prompt \
  --exp_name=$EXP5_NAME \
  --dataset=sketchy_ext \
  --data_dir=../data/Sketchy_kag \
  --gpt_text_file='gpt_file/sketchy_ext.json' \
  --n_ctx=2 \
  --prompt_depth=12 \
  --language_depth=1 \
  --cross_layer=6 \
  --lambda_cross_modal=1.0 \
  --lambda_consistency=1.0 \
  --lambda_ce=1.0 \
  --clip_LN_lr=1e-6 \
  --prompt_lr=1e-4 \
  --batch_size=128 \
  --workers=4 \
  --test_batch_size=1024 \
  --epochs=6 \
  --enhance_text \
  --lambda_text_consistency=2.0 \
  --lambda_visual_cross=0.2

# =================================================================
# THÍ NGHIỆM 6: 
# =================================================================
EXP1_NAME="clipat_hicropl_6_add_enhancetext"
echo ">>> [1/5] Đang chạy thí nghiệm: $EXP1_NAME ..."
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.hicropl_prompt \
  --exp_name=$EXP1_NAME \
  --dataset=sketchy_ext \
  --data_dir=../data/Sketchy_kag \
  --n_ctx=2 \
  --prompt_depth=12 \
  --language_depth=1 \
  --cross_layer=6 \
  --lambda_cross_modal=1.0 \
  --lambda_consistency=1.0 \
  --lambda_ce=1.0 \
  --clip_LN_lr=1e-6 \
  --prompt_lr=1e-4 \
  --batch_size=128 \
  --workers=4 \
  --test_batch_size=1024 \
  --epochs=6 \
  --disable_cross_exchange \
  --disable_augmentation \
  --enhance_text \
  --lambda_text_consistency=2.0 \
  --lambda_visual_cross=0.2

echo ">>> Thí nghiệm $EXP1_NAME ĐÃ XONG!"
rm -rf "saved_models/"
echo "------------------------------------------------------"


# =================================================================
# THÍ NGHIỆM 7: 
# =================================================================
EXP2_NAME="clipat_hicropl_7_add_only_augmentation"
echo ">>> [2/5] Đang chạy thí nghiệm: $EXP2_NAME ..."
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.hicropl_prompt \
  --exp_name=$EXP2_NAME \
  --dataset=sketchy_ext \
  --data_dir=../data/Sketchy_kag \
  --n_ctx=2 \
  --prompt_depth=12 \
  --language_depth=1 \
  --cross_layer=6 \
  --lambda_cross_modal=1.0 \
  --lambda_consistency=1.0 \
  --lambda_ce=1.0 \
  --clip_LN_lr=1e-6 \
  --prompt_lr=1e-4 \
  --batch_size=128 \
  --workers=4 \
  --test_batch_size=1024 \
  --epochs=6 \
  --disable_cross_exchange

echo ">>> Thí nghiệm $EXP2_NAME ĐÃ XONG!"
rm -rf "saved_models/"
echo "------------------------------------------------------"

echo ">>> Thí nghiệm $EXP5_NAME ĐÃ XONG!"
rm -rf "saved_models/"
echo "======================================================"
echo "CHÚC MỪNG! TOÀN BỘ 5 THÍ NGHIỆM ABLATION ĐÃ HOÀN THÀNH."
echo "Kết quả Tensorboard đã được lưu lại trong tb_logs."
