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

# ----------------------
# ViT & HiCroPL Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

# HiCroPL Params
parser.add_argument('--n_ctx', type=int, default=4, help='Number of context tokens for prompts')
parser.add_argument('--prompt_depth', type=int, default=9, help='Depth of deep prompts')
parser.add_argument('--cross_layer', type=int, default=-1, help='Layer boundary k for the photo<->sketch exchange (shallow=Photo->Sketch, deep=Sketch->Photo); -1 means prompt_depth // 2')
parser.add_argument('--disable_exchange', action='store_true', help='Ablation: disable the photo<->sketch cross-domain exchange (LKP+Mapper) while keeping everything else (k-means init, n_ctx, prompt_depth, LR) identical, so cross_prompts_photo/sketch train as fully independent per-branch prompts. Isolates the exchange mechanism as the only variable between paired ON/OFF runs.')
parser.add_argument('--exchange_detach_source', action='store_true', help='Ablation (branch A): in the Photo->Sketch mapping block, detach the photo proxy tokens (P~_photo, the attn_pooling_photo output) before feeding them as k/v to photo2sketch_net. Numeric values are unchanged; only the gradient path from the sketch-side loss back into ctx_photo/attn_pooling_photo_nets is cut. Isolates whether the cross-exchange gradient feedback into the photo branch matters. Mutually exclusive with --exchange_free_source.')
parser.add_argument('--exchange_free_source', action='store_true', help='Ablation (branch B): in the Photo->Sketch mapping block, replace the photo proxy tokens (P~_photo) fed as k/v to photo2sketch_net with an independent nn.Parameter (same shape, normal std=0.02 init, no k-means, unrelated to any photo feature). Mapper param count is unchanged. Isolates whether photo-derived content in the proxy matters, or any learnable source suffices. Mutually exclusive with --exchange_detach_source and --exchange_self_source.')
parser.add_argument('--exchange_self_source', action='store_true', help='Ablation (control experiment): in the Photo->Sketch mapping block, feed cross_prompts_sketch[i] (not cross_prompts_photo[i]) into attn_pooling_photo_nets[i] -- same LKP module, same photo2sketch_net Mapper, same query, same overwrite of current_sketch_prompts[i], proxy detached before the Mapper exactly like --exchange_detach_source. No photo tensor participates in this block at all; sketch self-refines through the identical pipeline. Isolates whether Photo->Sketch benefit comes from the source being PHOTO specifically, vs. just having a (detached) same-shape source flow through this pipeline. No parameters added or removed vs. --exchange_detach_source. Mutually exclusive with --exchange_detach_source and --exchange_free_source.')
parser.add_argument('--mapper_single_scale', action='store_true', help='Ablation (control experiment, reproduces the original HiCroPL paper\'s Table 6 single-scale vs multi-scale comparison): in the Photo->Sketch mapping block, the proxy computation (LKP over cross_layer layers, same attn_pooling_photo_nets, same detach behavior of the other exchange_* flags) is unchanged, but photo2sketch_net at layer i is restricted to key/value = only that layer\'s own proxy p~^i (shape [1, dim]) instead of the full concatenated set of all cross_layer proxies (shape [cross_layer, dim]). Isolates whether the block\'s benefit comes from each layer seeing every other layer\'s proxy (multi-scale/hierarchical), or just from passing through one more learned module. Orthogonal to --exchange_self_source/--exchange_detach_source/--exchange_free_source (freely combinable, no mutual exclusion). Adds/removes no parameters or modules.')
parser.add_argument('--no_prompt_learning', action='store_true', help='Ablation: disable ALL prompt learning (no visual or text prompt tokens at all -- forces clip_trainer to a vanilla, non-prompted CLIP). Only LayerNorm stays trainable (CLIP-AT-style baseline), text uses the fixed ctx_init/ctx_init_sketch template with no learnable context. Mutually exclusive in spirit with --disable_exchange (this is a strictly more minimal baseline).')
parser.add_argument('--use_text_visual_exchange', action='store_true', help='Alternative architecture: per-branch bidirectional text<->visual prompt exchange (LKP+Mapper), one independent pair for photo and one for sketch -- NO photo<->sketch coupling at all (mutually exclusive with the default photo<->sketch VisualVisualPromptLearner architecture). --disable_exchange has NO effect when this is set (it only applies to the photo<->sketch architecture); use the disable_exchange=True baseline as the shared no-exchange reference for comparing both architectures.')
parser.add_argument('--ctx_init', type=str, default='a photo of a', help='Initial text context for photo prompt learner')
parser.add_argument('--ctx_init_sketch', type=str, default='a sketch of a', help='Initial text context for sketch prompt learner')
parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for InfoNCE Loss')
parser.add_argument('--lambda_cross_modal', type=float, default=1.0, help='Weight for sketch-photo InfoNCE alignment')
parser.add_argument('--lambda_ce', type=float, default=1.0, help='Weight for prompted visual-text classification loss')
parser.add_argument('--kmeans_min_per_category', type=int, default=3, help='Min photo images per category sampled (stratified) for k-means prompt-init anchor')

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
