import argparse

parser = argparse.ArgumentParser(description='Sketch-based OD')

parser.add_argument('--exp_name', type=str, default='LN_prompt')

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
parser.add_argument('--freeze_text', action='store_true',
                    help='Freeze the ENTIRE text branch (text transformer + token/positional embedding + ln_final + text_projection), including its LayerNorm. Visual branch still trains LN + prompts.')

# Patch shuffle options (self-supervised auxiliary loss)
# (patch-shuffle handled in FG training code; no CLI options required)

# ----------------------
# ViT & HiCroPL Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

# CLIP-AT baseline params (Sain et al. CVPR'23)
parser.add_argument('--n_ctx', type=int, default=3, help='Number of visual prompt tokens (CLIP-AT: K=3)')
parser.add_argument('--ctx_init', type=str, default='a photo of a', help='Hard text template for the photo modality (replaced by learnable text_prompt_photo)')
parser.add_argument('--ctx_init_sketch', type=str, default='a sketch of a', help='Hard text template for the sketch modality (replaced by learnable text_prompt_sketch)')
parser.add_argument('--triplet_margin', type=float, default=0.2, help='Margin for triplet loss (CLIP-AT official: 0.2)')
parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for classification softmax')
parser.add_argument('--lambda_cross_modal', type=float, default=1.0, help='Weight for sketch-photo alignment loss (triplet or InfoNCE)')
parser.add_argument('--cross_modal_loss', type=str, default='triplet', choices=['infonce', 'triplet'], help='Sketch-photo alignment: triplet (CLIP-AT, default) or InfoNCE (in-batch negatives)')
parser.add_argument('--text_prompt_mode', type=str, default='template', choices=['template', 'learnable'], help="Text branch: 'template' (CLIP-AT hard template, default) or 'learnable' (CoOp context tokens)")
parser.add_argument('--lambda_ce', type=float, default=0.5, help='Weight for visual-text classification loss (CLIP-AT lambda1)')
parser.add_argument('--lambda_consistency', type=float, default=1.0, help='Weight for visual distill consistency loss')
parser.add_argument('--lambda_text_consistency', type=float, default=1.0, help='Weight for GPT text distill consistency loss')
parser.add_argument('--gpt_text_file', type=str, default='gpt_file/sketchy_ext.json', help='GPT text prompt JSON for modality-specific distill text branches')

# CLIP design_details (CoPrompt-style builder config)
parser.add_argument('--clip_trainer', type=str, default='HiCroPL', help='Trainer key for CLIP block routing')
parser.add_argument('--vision_depth', type=int, default=-1, help='Prompted visual depth; -1 means use prompt_depth')
parser.add_argument('--language_depth', type=int, default=-1, help='Prompted text depth; -1 means use prompt_depth')
parser.add_argument('--vision_ctx', type=int, default=-1, help='Visual prompt token count; -1 means use n_ctx')
parser.add_argument('--language_ctx', type=int, default=-1, help='Text prompt token count; -1 means use n_ctx')

# ----------------------
# Evaluation Mode
# ----------------------
parser.add_argument('--eval_mode', type=str, default='category', 
                    choices=['category', 'fine_grained'],
                    help='Evaluation mode: category-level retrieval or fine-grained instance-level retrieval')

opts = parser.parse_args()
