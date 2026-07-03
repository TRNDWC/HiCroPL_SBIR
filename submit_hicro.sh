#!/bin/bash
#SBATCH --job-name=hicropl_ablated
#SBATCH --output=logs/hicropl_ablated.out
#SBATCH --error=logs/hicropl_ablated.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00

eval "$(conda shell.bash hook)"
module load Anaconda3
source activate
conda activate hicropl
export PYTHONNOUSERSITE=True    # prevent using packages from base

# Chuyển vào thư mục chứa code để script python có thể tìm thấy module experiments
cd /home/hoangu/HiCroPL_SBIR || exit

CUDA_VISIBLE_DEVICES=0 python -u -m experiments.hicropl_prompt \
  --exp_name=clipat_hicropl_enhance_w_lp1_upd \
  --dataset=sketchy_ext \
  --data_dir=../data/Sketchy \
  --gpt_text_file='gpt_file/sketchy_ext.json' \
  --n_ctx=2 \
  --prompt_depth=12 \
  --language_depth=1 \
  --cross_layer=6 \
  --lambda_cross_modal=1.0 \
  --lambda_consistency=1.0 \
  --lambda_ce=1.5 \
  --clip_LN_lr=1e-5 \
  --prompt_lr=1e-5 \
  --batch_size=128 \
  --workers=4 \
  --test_batch_size=1024 \
  --epochs=60 \
  --enhance_text \
  --lambda_text_consistency=3.0 \
  --lambda_visual_cross=0.5
