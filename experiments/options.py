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
parser.add_argument('--backbone', type=str, default='ViT-B/32', help='CLIP backbone name')

# ----------------------
# ViT & HiCroPL Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

# HiCroPL Params
parser.add_argument('--n_ctx', type=int, default=4, help='Number of context tokens for prompts')
parser.add_argument('--prompt_depth', type=int, default=9, help='Depth of deep prompts')
parser.add_argument('--cross_layer', type=int, default=4, help='Layer at which bidirectional flow switches direction')
parser.add_argument('--ctx_init', type=str, default='a photo or a sketch of a', help='Initial text context for photo prompt learner')
parser.add_argument('--ctx_init_sketch', type=str, default='a photo or a sketch of a', help='Initial text context for sketch prompt learner')
parser.add_argument('--lambda_triplet', type=float, default=1.0, help='Weight for L1: Triplet Loss')
parser.add_argument('--lambda_cross_modal', type=float, default=1.0, help='Weight for L2: InfoNCE (sketch-photo)')
parser.add_argument('--lambda_consistency', type=float, default=1.0, help='Weight for L3: InfoNCE (augmentation consistency)')
parser.add_argument('--lambda_ce', type=float, default=1.0, help='Weight for L4: Cross-Entropy Loss')
parser.add_argument('--lambda_ce_aug', type=float, default=0.0, help='Weight for L5: Cross-Entropy Loss (augmented), disabled by default for CoPrompt-style flow')
parser.add_argument('--lambda_text_align', type=float, default=0.1, help='Weight for text-branch alignment InfoNCE between photo/sketch text features')
parser.add_argument('--lambda_mcc_sk', type=float, default=0.0, help='Weight for L6: MCC Loss (sketch intra-modal), disabled by default for CoPrompt-style flow')
parser.add_argument('--lambda_mcc_ph', type=float, default=0.0, help='Weight for L6: MCC Loss (photo intra-modal), disabled by default for CoPrompt-style flow')
parser.add_argument('--mcc_sk', type=float, default=0.1, help='Target mean similarity for sketch-to-sketch (MCC center)')
parser.add_argument('--mcc_ph', type=float, default=0.0, help='Target mean similarity for photo-to-photo (MCC center)')
parser.add_argument('--triplet_margin', type=float, default=0.3, help='Margin for Triplet Loss')
parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for InfoNCE Loss')
parser.add_argument('--adapter_reduction', type=int, default=4, help='Bottleneck reduction ratio for CoPrompt-style adapters')
parser.add_argument('--image_adapter_m', type=float, default=0.1, help='Deprecated: residual mixing is disabled')
parser.add_argument('--visual_adapter_m', type=float, default=0.1, help='Deprecated alias; residual mixing is disabled')
parser.add_argument('--text_adapter_m', type=float, default=0.1, help='Deprecated: residual mixing is disabled')
parser.add_argument('--lambda_distill', type=float, default=1.0, help='Weight for pairwise cosine distillation between base and augmented features')

# CLIP design_details (CoPrompt-style builder config)
parser.add_argument('--clip_trainer', type=str, default='HiCroPL', help='Trainer key for CLIP block routing')
parser.add_argument('--vision_depth', type=int, default=-1, help='Prompted visual depth; -1 means use prompt_depth')
parser.add_argument('--language_depth', type=int, default=-1, help='Prompted text depth; -1 means use prompt_depth')
parser.add_argument('--vision_ctx', type=int, default=-1, help='Visual prompt token count; -1 means use n_ctx')
parser.add_argument('--language_ctx', type=int, default=-1, help='Text prompt token count; -1 means use n_ctx')

opts = parser.parse_args()
