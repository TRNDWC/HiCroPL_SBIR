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
# ViT & Baseline Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)


# ----------------------
# Evaluation Mode
# ----------------------
parser.add_argument('--eval_mode', type=str, default='category', 
                    choices=['category', 'fine_grained'],
                    help='Evaluation mode: category-level retrieval or fine-grained instance-level retrieval')

opts = parser.parse_args()
