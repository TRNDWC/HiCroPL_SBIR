# 1. Đảm bảo đã cài công cụ tải Google Drive
pip install gdown

# 2. Tạo thư mục data (nếu chưa có)
mkdir -p data

# 3. Tải file Sketchy từ Google Drive (Dung lượng khoảng 2.8GB)
gdown 1vGtssYgM6_r0ph8f_ZPWzIHvHL0yS8CN -O data/Sketchy.zip

# 4. Giải nén vào thư mục data/
unzip data/Sketchy.zip -d data/

``
python -m experiments.hicropl_prompt --exp_name=hicropl_prompt 
--n_prompts=3 
--clip_LN_lr=1e-6 
--prompt_lr=5e-4 
--batch_size=256 
--workers=8 
--data_dir="../data/Sketchy"

python -m experiments.hicropl_prompt \
  --exp_name=clipat_paper \
  --dataset=sketchy_ext \
  --data_dir="../data/Sketchy"  \
  --n_ctx=3 \
  --cross_modal_loss=triplet \
  --triplet_margin=0.3 \
  --text_prompt_mode=template \
  --lambda_cross_modal=1.0 \
  --lambda_ce=0.5 \
  --clip_LN_lr=1e-5 \
  --prompt_lr=1e-4 \
  --batch_size=64 \
  --workers=8 \
  --weight_decay=0.0 \
  --test_batch_size=1024 \
  --epochs=15
  --

python -m experiments.hicropl_prompt \
  --exp_name=dual_desc_hicropl \
  --dataset=sketchy_ext \
  --data_dir=../data/Sketchy \
  --n_ctx=3 \
  --cross_modal_loss=triplet \
  --triplet_margin=0.3 \
  --text_prompt_mode=learnable \
  --gpt_text_file=gpt_file/gemini_sketchy_ext.json \
  --lambda_cross_modal=1.0 \
  --lambda_ce=0.5 \
  --lambda_text_consistency=1.0 \
  --lambda_consistency=1.0 \
  --clip_LN_lr=1e-5 \
  --prompt_lr=1e-5 \
  --batch_size=64 \
  --weight_decay=0.0 \
  --test_batch_size=1024 \
  --epochs=15

!python -m experiments.hicropl_prompt \
  --exp_name=dual_desc_hicropl \
  --dataset=sketchy_ext \
  --data_dir=../data/Sketchy \
  --n_ctx=3 \
  --cross_modal_loss=triplet \
  --triplet_margin=0.3 \
  --text_prompt_mode=learnable \
  --gpt_text_file=gpt_file/gemini_sketchy_ext.json \
  --lambda_cross_modal=1.0 \
  --lambda_ce=0.5 \
  --lambda_text_consistency=1.0 \
  --lambda_consistency=1.0 \
  --warmup_epochs=5 \
  --clip_LN_lr=1e-5 \
  --prompt_lr=1e-5 \
  --batch_size=64 \
  --weight_decay=0.0 \
  --test_batch_size=1024 \
  --epochs=15
