import argparse

parser = argparse.ArgumentParser(description='Sketch-based OD')

parser.add_argument('--exp_name', type=str, default='LN_prompt')
parser.add_argument('--seed', type=int, default=42,
                    help='Seed cho Python/NumPy/PyTorch/CUDA và DataLoader worker. '
                         'Đổi seed để ước lượng dao động giữa các lần chạy')

# --------------------
# DataLoader Options
# --------------------

# Path to 'Sketchy' folder holding Sketch_extended dataset. It should have 2 folders named 'sketch' and 'photo'.
parser.add_argument('--dataset', type=str, default='sketchy', 
                    choices=['sketchy', 'sketchy_ext', 'tuberlin', 'quickdraw'],
                    help='Dataset name: sketchy, sketchy_ext, tuberlin, or quickdraw')
parser.add_argument('--data_dir', type=str, default='/isize2/sain/data/Sketchy/') 
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--data_split', type=float, default=-1.0)

# ----------------------
# Training Params
# ----------------------

parser.add_argument('--clip_lr', type=float, default=1e-4)
# Mặc định 1e-4 = đúng giá trị fallback mà configure_optimizers vẫn dùng trước
# khi có option này, nên thêm vào không đổi hành vi của bất kỳ run nào.
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay của Adam, áp cho cả nhóm prompt lẫn nhóm LayerNorm')
parser.add_argument('--clip_LN_lr', type=float, default=1e-5)
parser.add_argument('--prompt_lr', type=float, default=1e-5)
parser.add_argument('--linear_lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--backbone', type=str, default='ViT-B/32', 
                    choices=['ViT-B/32'], 
                    help='CLIP backbone name')
parser.add_argument('--num_trainable_ln', type=int, default=-1, 
                    help='Number of LayerNorm layers to train (counting from the end). -1 means all.')

# Patch shuffle options (self-supervised auxiliary loss)
# (patch-shuffle handled in FG training code; no CLI options required)

# ----------------------
# Ablation Study Flags
# ----------------------
parser.add_argument('--disable_cross_exchange', action='store_true', help='Disable Visual-Visual cross-modal token exchange')
parser.add_argument('--disable_augmentation', action='store_true', help='Disable data augmentation and consistency loss')
parser.add_argument('--enhance_text', action='store_true', help='Enable L3 and L_cons_visual_cross for text enhancement')

# ----------------------
# ViT & HiCroPL Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

# HiCroPL Params
parser.add_argument('--n_ctx', type=int, default=4, help='Number of context tokens for prompts')
parser.add_argument('--prompt_depth', type=int, default=9, help='Depth of deep prompts')
parser.add_argument('--cross_layer', type=int, default=4, help='Layer at which bidirectional flow switches direction')
parser.add_argument('--ctx_init', type=str, default='a photo of a', help='Initial text context for photo prompt learner')
parser.add_argument('--ctx_init_sketch', type=str, default='a sketch of a', help='Initial text context for sketch prompt learner')
parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for InfoNCE Loss')
parser.add_argument('--lambda_cross_modal', type=float, default=1.0, help='Weight for sketch-photo InfoNCE alignment')
parser.add_argument('--lambda_ce', type=float, default=1.0, help='Weight for prompted visual-text classification loss')
parser.add_argument('--lambda_consistency', type=float, default=1.0, help='Weight for visual distill consistency loss')
parser.add_argument('--lambda_text_consistency', type=float, default=1.0, help='Weight for GPT text distill consistency loss')
parser.add_argument('--lambda_visual_cross', type=float, default=0.1, help='Weight for cross-anchor visual to text loss')
parser.add_argument('--gpt_text_file', type=str, default='gpt_file/sketchy_ext.json', help='GPT text prompt JSON for modality-specific distill text branches')

# CLIP design_details (CoPrompt-style builder config)
parser.add_argument('--clip_trainer', type=str, default='HiCroPL', help='Trainer key for CLIP block routing')
parser.add_argument('--vision_depth', type=int, default=-1, help='Prompted visual depth; -1 means use prompt_depth')
parser.add_argument('--language_depth', type=int, default=-1, help='Prompted text depth; -1 means use prompt_depth')
parser.add_argument('--vision_ctx', type=int, default=-1, help='Visual prompt token count; -1 means use n_ctx')
parser.add_argument('--language_ctx', type=int, default=-1, help='Text prompt token count; -1 means use n_ctx')

# ----------------------
# Checkpoint & Logging
# ----------------------
parser.add_argument('--save_dir', type=str, default='saved_models',
                    help='Thư mục gốc chứa checkpoint. Checkpoint đi vào <save_dir>/<exp_name>/')
parser.add_argument('--log_dir', type=str, default='tb_logs',
                    help='Thư mục gốc cho TensorBoard logs. Logs đi vào <log_dir>/<exp_name>/')
parser.add_argument('--save_top_k', type=int, default=1,
                    help='Số checkpoint tốt nhất giữ lại theo metric monitor. -1 = giữ tất cả, 0 = không lưu')
parser.add_argument('--save_last', action='store_true',
                    help='Lưu thêm last.ckpt sau mỗi epoch. BẮT BUỘC bật nếu muốn auto-resume')
parser.add_argument('--no_resume', action='store_true',
                    help='Bỏ qua last.ckpt có sẵn, luôn train từ đầu')
parser.add_argument('--summary_csv', type=str, default='runs_summary.csv',
                    help='CSV tổng hợp 1 dòng/run, NẰM NGOÀI log_dir để so sánh các run')
parser.add_argument('--log_every_n_steps', type=int, default=50,
                    help='Ghi một dòng train_steps.csv mỗi N step. 0 = tắt log theo step')
parser.add_argument('--run_id', type=str, default='',
                    help='Định danh run (mặc định là timestamp). Quyết định thư mục '
                         '<log_dir>/<exp_name>/<run_id>/ chứa log riêng của lần chạy này')

# ----------------------
# Verification
# ----------------------
parser.add_argument('--learn_logit_scale', action='store_true',
                    help='Cho phép học logit_scale (clamp <= log(100)). Mặc định đóng băng theo tinh thần prompt tuning')

# ----------------------
# Chẩn đoán
# ----------------------
parser.add_argument('--eval_frozen_only', action='store_true',
                    help='Bỏ hoàn toàn nhánh prompted khi eval, chỉ dùng đặc trưng CLIP '
                         'đóng băng. Đo trần dưới: prompt thực sự đóng góp bao nhiêu điểm')

# ----------------------
# Evaluation Mode
# ----------------------
parser.add_argument('--eval_mode', type=str, default='category', 
                    choices=['category', 'fine_grained'],
                    help='Evaluation mode: category-level retrieval or fine-grained instance-level retrieval')

opts = parser.parse_args()
