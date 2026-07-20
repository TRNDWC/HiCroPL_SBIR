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

# Patch shuffle options (self-supervised auxiliary loss)
# (patch-shuffle handled in FG training code; no CLI options required)

# ----------------------
# ViT & HiCroPL Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

# HiCroPL Params
parser.add_argument('--n_ctx', type=int, default=4, help='Number of context tokens for prompts')
parser.add_argument('--vision_depth', type=int, default=1, help='Number of ViT layers with learnable prompt injection')
parser.add_argument('--text_depth', type=int, default=1, help='Number of text transformer layers with learnable prompt injection')
parser.add_argument('--cross_layer', type=int, default=4, help='Layer at which bidirectional flow switches direction')
parser.add_argument('--ctx_init', type=str, default='a photo of a', help='Initial text context for photo prompt learner')
parser.add_argument('--ctx_init_sketch', type=str, default='a sketch of a', help='Initial text context for sketch prompt learner')
parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for InfoNCE Loss')
parser.add_argument('--lambda_cross_modal', type=float, default=1.0, help='Weight for sketch-photo InfoNCE alignment')
parser.add_argument('--lambda_ce', type=float, default=1.0, help='Weight for prompted visual-text classification loss')
parser.add_argument('--use_content_cond', action='store_true', help='[H2] Condition sketch prompt on this batch\'s photo embedding via cross-attention gate (photo branch unchanged)')
parser.add_argument('--content_dropout_prob', type=float, default=0.4, help='[H2] Probability of withholding the photo descriptor during training (modality dropout), so the gate copes with the no-photo case at retrieval time')

# CLIP design_details (CoPrompt-style builder config)
parser.add_argument('--clip_trainer', type=str, default='HiCroPL', help='Trainer key for CLIP block routing')

# ----------------------
# Evaluation Mode
# ----------------------
parser.add_argument('--eval_mode', type=str, default='category', 
                    choices=['category', 'fine_grained'],
                    help='Evaluation mode: category-level retrieval or fine-grained instance-level retrieval')

opts = parser.parse_args()
