!python -m experiments.hicropl_prompt \
  --exp_name=clipat_paper_new_layernorm_new_dataset \
  --dataset=sketchy_ext \
  --data_dir=../data/Sketchy \
  --n_ctx=3 \
  --lambda_cross_modal=1 \
  --lambda_ce=0.5 \
  --clip_LN_lr=1e-5 \
  --prompt_lr=1e-5 \
  --batch_size=16 \
  --test_batch_size=1024 \
  --epochs=60

python -m experiments.hicropl_prompt \
  --exp_name=clipat_paper_dual_descriptions \
  --dataset=sketchy_ext \
  --data_dir=../data/Sketchy \
  --n_ctx=3 \
  --lambda_cross_modal=1 \
  --lambda_ce=0.5 \
  --clip_LN_lr=1e-5 \
  --prompt_lr=1e-5 \
  --batch_size=128 \
  --workers=2 \
  --test_batch_size=1024 \
  --epochs=60 \
  --lambda_text_consistency=1.0 \
  --lambda_visual_cross=0.1
  # --gpt_text_file="gpt_file/gemini_sketchy_ext copy.json"

python -m experiments.hicropl_prompt \
--exp_name=clipat_paper_dual_descriptions \
--dataset=sketchy_ext \
--data_dir=../data/Sketchy \
--n_ctx=3 \
--lambda_cross_modal=1 \
--lambda_ce=0.1 \
--clip_LN_lr=1e-5 \
--prompt_lr=1e-5 \
--batch_size=64 \
--workers=2 \
--test_batch_size=1024 \
--epochs=60 \
--lambda_text_consistency=0 \
--lambda_visual_cross=0
# --gpt_text_file="gpt_file/gemini_sketchy_ext copy.json"

python -m experiments.hicropl_prompt \
  --exp_name=max_setting_point1_v3 \
  --dataset=sketchy_ext \
  --data_dir=../data/Sketchy \
  --n_ctx=2 \
  --prompt_depth=12 \
  --cross_layer=6 \
  --lambda_cross_modal=1 \
  --lambda_ce=1 \
  --clip_LN_lr=1e-6 \
  --prompt_lr=1e-6 \
  --batch_size=64 \
  --workers=4 \
  --test_batch_size=1024 \
  --epochs=15 \
  --lambda_text_consistency=1 \
  --lambda_visual_cross=0.5