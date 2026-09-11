import argparse

parser = argparse.ArgumentParser(description='Sketch-based OD')

parser.add_argument('--exp_name', type=str, default='LN_prompt')

# --------------------
# DataLoader Options
# --------------------

# Path to 'Sketchy' folder holding Sketch_extended dataset. It should have 2 folders named 'sketch' and 'photo'.
parser.add_argument('--dataset', type=str, default='sketchy_1',
                    choices=['sketchy_1', 'sketchy_2', 'tuberlin', 'quickdraw'],
                    help='Dataset name: sketchy_1, sketchy_2, tuberlin, or quickdraw')
parser.add_argument('--data_dir', type=str, default='/isize2/sain/data/Sketchy/')
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--data_split', type=float, default=-1.0)
parser.add_argument('--gzs_eval', action='store_true',
                    help='NON-STANDARD GZS eval (kept for backward compat, prefer --eval_mode_gzs): '
                         'mix a fixed, hand-picked SEEN-class subset '
                         '(GENERALIZED_CLASSES in src/dataset_retrieval.py, keyed by --dataset) '
                         'into the ValidDataset (sketch AND photo) gallery/query on top of the '
                         'unseen classes. Mutually exclusive with --eval_mode_gzs.')
parser.add_argument('--eval_mode_gzs', action='store_true',
                    help='Standard GZS-SBIR protocol: gallery = P^s (ALL seen-class train photos, '
                         'no subsampling) union P^u_test (the existing unseen-class photo gallery, '
                         'unchanged). Query stays S^u_test (unseen sketches only, unchanged) -- only '
                         'the photo gallery grows. Seen-class gallery images share no label with any '
                         'query, so they only ever act as distractors, never positives. Off by '
                         'default (identical to plain ZS-SBIR). Mutually exclusive with --gzs_eval '
                         'and --cross_dataset_eval.')
parser.add_argument('--eval_mode_gzs_ocean', action='store_true',
                    help='OCEAN (Zhu et al., ICME 2020, "Ocean: A Dual Learning Approach For '
                         'Generalized Zero-Shot Sketch-Based Image Retrieval") GZS-SBIR protocol -- '
                         'reproduces Table 1\'s "Test classes (GZS-SBIR)" counts exactly (Sketchy 30, '
                         'TU-Berlin 36). C^g = C^u union round(0.2 * |C^u|) randomly chosen WHOLE seen '
                         'classes (fixed seed 42, identical pick shared by the sketch and photo '
                         'instances); the test set D^g = {X^g, Y^g} is drawn from C^g for BOTH query '
                         '(sketch) and gallery (photo) -- unlike --eval_mode_gzs, seen-class sketches '
                         'ARE part of the query here, not just distractors in the gallery. Every image '
                         'of the selected extra seen classes is included (no per-image sampling). '
                         'Mutually exclusive with --eval_mode_gzs, --gzs_eval and --cross_dataset_eval.')
parser.add_argument('--eval_mode_gzs_drclip', action='store_true',
                    help='Dr. CLIP (Li et al., ACM MM 2024, "Dr. CLIP: CLIP-Driven Universal '
                         'Framework for Zero-Shot Sketch Image Retrieval") GZS-SBIR protocol -- '
                         'reproduces its Table 1 "Testing classes" exactly (Sketchy-G 42, '
                         'TU-Berlin-G 74). Same mechanism as --eval_mode_gzs_ocean (whole extra SEEN '
                         'classes join C^g; both query and gallery drawn from C^g) but the 20% is a '
                         'fraction of |C^s| (seen classes) instead of |C^u|: "the images of 20%% of '
                         'the seen categories in the training set were augmented into the test set". '
                         'Mutually exclusive with --eval_mode_gzs_ocean, --eval_mode_gzs, --gzs_eval '
                         'and --cross_dataset_eval.')
parser.add_argument('--gzs_seen_frac', type=float, default=1.0,
                    help='DEBUG ONLY, no effect unless --eval_mode_gzs is set. Fraction of P^s '
                         '(seen-class gallery photos) to actually load, sampled deterministically '
                         '(fixed seed) per seen class. Default 1.0 = full P^s, the correct GZS-SBIR '
                         'protocol -- this is the only setting whose numbers are valid to report or '
                         'compare. A value < 1.0 (e.g. 0.2) exists purely to speed up smoke-testing '
                         'the eval loop before a real run; mAP/P@k from such a run are NOT comparable '
                         'to any full-P^s result and must not be reported as GZS-SBIR numbers.')
parser.add_argument('--cross_dataset_eval', action='store_true',
                    help='Across-dataset ZS-SBIR: train on --dataset (e.g. sketchy) using ALL of '
                         'its categories (no within-dataset unseen holdout), and evaluate on a '
                         'different dataset entirely (--eval_dataset / --eval_data_dir), which is '
                         'fully unseen since its categories never appeared during training. '
                         'Mutually exclusive with --gzs_eval.')
parser.add_argument('--eval_dataset', type=str, default=None, choices=['tuberlin', 'quickdraw'],
                    help='Target dataset for --cross_dataset_eval. Selects the map_k/P@k '
                         'convention used for that dataset in on_validation_epoch_end.')
parser.add_argument('--eval_data_dir', type=str, default=None,
                    help='Root directory of --eval_dataset for --cross_dataset_eval (must contain '
                         '"sketch" and "photo" subfolders). Required when --cross_dataset_eval is set.')

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
parser.add_argument('--exchange_detach_source', action='store_true', help='Ablation (branch A): detach the SOURCE proxy tokens before they are fed as k/v to the Mapper, in whichever direction is active -- P~_photo before photo2sketch_net in the Photo->Sketch block, and P~_sketch before sketch2photo_net in the Sketch->Photo block. Symmetric on purpose: --cross_layer selects the direction (=prompt_depth for Photo->Sketch only, 0 for Sketch->Photo only, in between for both), and the flag must mean the same thing at every setting, otherwise the direction variants would differ by a whole LKP worth of trainable capacity instead of by direction. Numeric values are unchanged; only the gradient path from the sketch-side loss back into ctx_photo/attn_pooling_photo_nets is cut. Isolates whether the cross-exchange gradient feedback into the photo branch matters. Mutually exclusive with --exchange_free_source.')
parser.add_argument('--exchange_free_source', action='store_true', help='Ablation (branch B): in the Photo->Sketch mapping block, replace the photo proxy tokens (P~_photo) fed as k/v to photo2sketch_net with an independent nn.Parameter (same shape, normal std=0.02 init, no k-means, unrelated to any photo feature). Mapper param count is unchanged. Isolates whether photo-derived content in the proxy matters, or any learnable source suffices. Mutually exclusive with --exchange_detach_source and --exchange_self_source.')
parser.add_argument('--exchange_self_source', action='store_true', help='Ablation (control experiment): in the Photo->Sketch mapping block, feed cross_prompts_sketch[i] (not cross_prompts_photo[i]) into attn_pooling_photo_nets[i] -- same LKP module, same photo2sketch_net Mapper, same query, same overwrite of current_sketch_prompts[i], proxy detached before the Mapper exactly like --exchange_detach_source. No photo tensor participates in this block at all; sketch self-refines through the identical pipeline. Isolates whether Photo->Sketch benefit comes from the source being PHOTO specifically, vs. just having a (detached) same-shape source flow through this pipeline. No parameters added or removed vs. --exchange_detach_source. Mutually exclusive with --exchange_detach_source and --exchange_free_source.')
parser.add_argument('--mapper_single_scale', action='store_true', help='Ablation (control experiment, reproduces the original HiCroPL paper\'s Table 6 single-scale vs multi-scale comparison): in the Photo->Sketch mapping block, the proxy computation (LKP over cross_layer layers, same attn_pooling_photo_nets, same detach behavior of the other exchange_* flags) is unchanged, but photo2sketch_net at layer i is restricted to key/value = only that layer\'s own proxy p~^i (shape [1, dim]) instead of the full concatenated set of all cross_layer proxies (shape [cross_layer, dim]). Isolates whether the block\'s benefit comes from each layer seeing every other layer\'s proxy (multi-scale/hierarchical), or just from passing through one more learned module. Orthogonal to --exchange_self_source/--exchange_detach_source/--exchange_free_source (freely combinable, no mutual exclusion). Adds/removes no parameters or modules.')
parser.add_argument('--sketch_self_refine', action='store_true', help='Ablation (capacity-matched no-exchange control, Run C): requires --disable_exchange. cross_prompts_sketch[i] passed through the existing photo2sketch_net module: query=cross_prompts_sketch[i] (full gradient), key=value=cross_prompts_sketch[i].detach() (no LKP, no proxy, no normalization, k/v detached, nothing from photo at all), overwrites current_sketch_prompts[i], for i in range(cross_layer). Unlike --exchange_self_source (which still routes through the LKP/attn_pooling_photo_nets and a multi-layer proxy set), this skips both entirely -- photo2sketch_net becomes just one extra learned refinement block on sketch prompts, receiving real gradient via the query path (unlike plain --disable_exchange alone, where it is idle/dead weight). Adds/removes no parameters or modules. Mutually exclusive with --sketch_self_refine_ln.')
parser.add_argument('--sketch_self_refine_ln', action='store_true', help='Ablation (capacity-matched no-exchange control, Run D): requires --disable_exchange. Same pipeline as --sketch_self_refine (Run C) but adds one new LayerNorm module (self.ln_selfrefine, standard init) applied to the k/v side only before detaching: query=cross_prompts_sketch[i] (full gradient), key=value=LayerNorm(cross_prompts_sketch[i]).detach() (normalized, still no LKP/proxy/compression, still detached). Isolates whether normalizing the self-refine k/v matters on top of Run C. Adds exactly one nn.LayerNorm(s_dim) module/param set relative to --sketch_self_refine. Mutually exclusive with --sketch_self_refine.')
parser.add_argument('--proxy_init', type=str, default='randn', choices=['randn', 'small', 'mean'], help='Ablation (L1): init scheme for photo_proxy_token/sketch_proxy_token (the LKP query, AttentionPooling). Applies to both branches, every layer. "randn" (default) = torch.randn(1, dim) std~1.0, the pre-ablation behavior, kept as default for exact backward compat. "small" = nn.init.normal_(std=0.02), syncing scale with every other tensor in the system (independent draw per layer). "mean" = proxy token for layer l initialized to cross_prompts_photo[l].mean(dim=0, keepdim=True).clone() (resp. sketch) at init time -- data-derived from that layer\'s own prompt content, independent nn.Parameter (no shared storage/grad coupling). Does not change tensor shape by itself (still [1, dim] unless combined with --n_proxy). Orthogonal to --n_proxy.')
parser.add_argument('--n_proxy', type=int, default=1, help='Ablation (L2\'): number of proxy tokens the LKP (AttentionPooling) produces per layer (default 1 = original behavior). photo_proxy_token/sketch_proxy_token shape becomes [n_proxy, dim]; photo2sketch_net/sketch2photo_net then receive cross_layer * n_proxy tokens as k/v instead of cross_layer. n_ctx is unchanged. n_proxy == n_ctx means 1:1 (no compression) by design, not a bug. New proxy rows follow --proxy_init (for "mean", all n_proxy rows start equal, by design). Orthogonal to --proxy_init.')
parser.add_argument('--prompt_branch', type=str, default='both', choices=['both', 'text', 'image'],
                    help='Ablation of the prompt design -- which modality keeps LEARNABLE prompts. '
                         '"both" (default) = unchanged. "text" = text prompts only: vision_depth is '
                         'forced to 0 so the visual towers run genuinely prompt-free (no shallow VPT '
                         'tokens, no deep visual prompts) and VisualVisualPromptLearner is frozen, '
                         'which also makes the photo<->sketch exchange inert. "image" = image prompts '
                         'only: language_depth is forced to 0 (no deep text prompts) and the text '
                         'learners are frozen, so ctx stays at its ctx_init value -- which IS the '
                         'plain "a photo of a" template embedding -- reducing the text tower to '
                         'vanilla CLIP. Both learners are still CONSTRUCTED in every mode so the RNG '
                         'stream (and thus every other module init) is bit-identical across the three '
                         'settings; the frozen side is dropped from the optimizer by the requires_grad '
                         'filter, so no idle params appear. Combine with --disable_exchange for the '
                         '"Text/Image Prompt (w/o exchange)" cell.')
parser.add_argument('--aug_side', type=str, default='both', choices=['both', 'photo', 'sketch'],
                    help='Ablation of the augmentation branch -- which side keeps its InfoNCE term. '
                         '"both" (default) = unchanged, loss_aug = cross_loss(photo, photo_aug) + '
                         'cross_loss(sketch, sketch_aug). "photo"/"sketch" keep only that side: the '
                         'other augmented view is never encoded (one full ViT forward saved) and its '
                         'term is absent from the loss, not zero-weighted. The dataset still emits '
                         'both augmented tensors either way -- dropping one there would shift the '
                         'torch RNG stream and silently change the clean views. No effect under '
                         '--disable_aug_branch (nothing to take a side of).')
parser.add_argument('--no_prompt_learning', action='store_true', help='Ablation: disable ALL prompt learning (no visual or text prompt tokens at all -- forces clip_trainer to a vanilla, non-prompted CLIP). Only LayerNorm stays trainable (CLIP-AT-style baseline), text uses the fixed ctx_init/ctx_init_sketch template with no learnable context. Mutually exclusive in spirit with --disable_exchange (this is a strictly more minimal baseline).')
parser.add_argument('--use_text_visual_exchange', action='store_true', help='Alternative architecture: per-branch bidirectional text<->visual prompt exchange (LKP+Mapper), one independent pair for photo and one for sketch -- NO photo<->sketch coupling at all (mutually exclusive with the default photo<->sketch VisualVisualPromptLearner architecture). --disable_exchange has NO effect when this is set (it only applies to the photo<->sketch architecture); use the disable_exchange=True baseline as the shared no-exchange reference for comparing both architectures.')
parser.add_argument('--ctx_init', type=str, default='a photo of a', help='Initial text context for photo prompt learner')
parser.add_argument('--ctx_init_sketch', type=str, default='a sketch of a', help='Initial text context for sketch prompt learner')
parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for InfoNCE Loss')
parser.add_argument('--lambda_cross_modal', type=float, default=1.0, help='Weight for sketch-photo InfoNCE alignment')
parser.add_argument('--lambda_ce', type=float, default=1.0, help='Weight for prompted visual-text classification loss')
parser.add_argument('--lambda_aug', type=float, default=1.0, help='Weight for the augmentation-branch InfoNCE loss, i.e. lambda_aug * (InfoNCE(photo, photo_aug) + InfoNCE(sketch, sketch_aug)). One weight for both terms. Default 1.0 reproduces the behaviour from before this flag existed. Setting 0.0 zeroes the loss but STILL builds clip_aug, still runs its two forward passes, and still leaves its params in the optimizer receiving zero gradient -- use --disable_aug_branch instead for a clean removal (no second backbone, no augmented tensors from the dataset, no idle params in the log).')
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

parser.add_argument('--disable_aug_branch', action='store_true', help='Ablation: fully remove the augmentation branch, which is ON by default. The branch is a SECOND CLIP backbone (self.clip_aug), built vanilla (clip_trainer=CoOp, no prompt injection) and frozen except for the LayerNorms of its VISUAL tower (39,936 params, opened at src/model_hicropl.py:456 and trained by the aug InfoNCE terms -- the forward is deliberately not under no_grad). It encodes an augmented view of the photo and of the sketch, and two InfoNCE terms, jointly weighted by --lambda_aug (default 1.0), pull the main prompted features toward those targets. Setting this flag is a CLEAN ablation in the same sense as --disable_exchange: clip_aug is never constructed, the dataset stops emitting the two augmented tensors, and the two loss terms are skipped -- nothing is built-then-idled. Note the branch costs ~605MB VRAM and ~151M params when enabled.')

# ----------------------
# Augmentation-branch decomposition (Run A / Run B)
#
# The default aug branch confounds TWO variables: (i) a second encoder exists
# (clip_aug: its own 151M weights, its own 39,936 trainable visual LayerNorms),
# and (ii) that encoder is fed a PERTURBED view. These two flags separate them,
# one variable each. Both default False -- every pre-existing run reproduces
# bit-identically. They are mutually exclusive (see the check below).
# ----------------------
parser.add_argument('--aug_shared_encoder', action='store_true', help='Run A -- isolates variable (ii), the augmented view, by deleting variable (i). clip_aug is NEVER constructed (no load_clip_to_cpu, so 151M params and ~605MB VRAM are saved and nothing of it can reach the optimizer or the checkpoint). The augmented views instead go through the EXACT main forward path: same backbone (self.clip), same visual_encoder_photo/sketch, same prompt tensors computed once in that same forward, same LayerNorms -- only the input tensor changes to photo_aug/sketch_aug. loss_aug keeps its exact structure and its 1.0 coefficient: cross_loss(photo_feat, photo_aug_feat, T) + cross_loss(sketch_feat, sketch_aug_feat, T). Mutually exclusive with --aug_identity_transform.')
parser.add_argument('--aug_detach_view', action='store_true', help='Sub-flag of --aug_shared_encoder (no effect otherwise): .detach() the augmented-view features before cross_loss, turning the term into a one-way pull of the clean view toward a fixed augmented target. Default OFF = symmetric siamese, which is what the current clip_aug branch actually does (its visual LayerNorms receive gradient from loss_aug -- there is no no_grad on that path, src/model_hicropl.py:542-562), so OFF is the setting that keeps Run A comparable to the 80.13 reference.')
parser.add_argument('--allow_degenerate_aug', action='store_true', help='Run D -- escape hatch that permits --aug_shared_encoder together with --aug_identity_transform, a combination otherwise rejected. It is NOT a no-op: photo_aug_feat comes out bit-identical to photo_feat, so the InfoNCE positive term is saturated (cos=1, nothing left to align), but the NEGATIVE term still pushes the batch apart -- the aug term degenerates into a pure intra-domain uniformity regularizer with the alignment half switched off. That separates the two things loss_aug does at once, which no other flag combination can. Off by default; the ValueError stays exactly as before without it.')
parser.add_argument('--aug_identity_transform', action='store_true', help='Run B -- isolates variable (i), the second encoder, by neutralizing variable (ii). clip_aug is still built, still gets its visual LayerNorms opened, still runs its two forward passes, and loss_aug keeps its exact structure -- but the augmented transform for BOTH photo and sketch is replaced by Sketchy.data_transform (the very transform used for the clean view), so photo_aug is bit-wise equal to photo and sketch_aug to sketch. Verified once on the first batch with torch.allclose (prints IDENTITY TRANSFORM: OK). Any gain that survives here comes from the second encoder itself, not from the perturbation. Mutually exclusive with --aug_shared_encoder.')

# ----------------------
# Class descriptions (VLM-generated, per-domain) -- see gpt_file/*.json
# Both must be supplied together; supplying neither keeps the hard-coded
# ctx_init/ctx_init_sketch template path untouched.
# ----------------------
parser.add_argument('--desc_sketch', type=str, default=None, help='Path to descriptions_sketch.json: {"meta": {...}, "descriptions": {classname: text}}. Replaces the hard-coded sketch template for the text branch. Must be given together with --desc_photo.')
parser.add_argument('--desc_photo', type=str, default=None, help='Path to descriptions_photo.json: {"meta": {...}, "descriptions": {classname: text}}. Replaces the hard-coded photo template for the text branch. Must be given together with --desc_sketch.')
parser.add_argument('--desc_pos', type=str, default='V1', choices=['V1', 'V2'], help='Where the class name sits relative to the description in the text prompt. V1 (default): "X..X <description>, a <class>." -- class last. V2: "X..X a <class>, <description>." -- class first, description last. The leading X tokens are placeholders the learnable ctx overwrites in both cases. V3 (class name inside the description) is not offered: the shipped description files have the class name stripped, so it cannot be built from them. No effect unless --desc_sketch/--desc_photo are given.')
parser.add_argument('--text_variant', type=str, default='template', choices=['template', 'desc_only', 'desc_sep', 'desc_shared'], help='Architecture of the text branch. Fully independent of every aug flag. "template" (default) = current baseline: only "a photo of a <class>." with learnable ctx feeds L_ce, nothing else is built and loss_text does not exist. "desc_only" = the VLM description REPLACES the template in L_ce ("x..x <desc>, a <class>."), still one sequence, still no loss_text -- the cheapest variant. "desc_sep" = template feeds L_ce and the description is a SECOND sequence encoded by a vanilla prompt-free text tower; loss_text pulls the two together. "desc_shared" = same as desc_sep except the description rides the MAIN clip with the same ctx and deep prompts, which doubles the text batch of that encoder. desc_sep and desc_shared differ in exactly one thing: which encoder sees the auxiliary sequence.')
parser.add_argument('--lambda_text', type=float, default=1.0, help='Weight of loss_text, the InfoNCE between the L_ce text feature and the auxiliary description feature. Only used by --text_variant desc_sep / desc_shared; the other two variants have no such term at all (not a zero-weighted one).')

opts = parser.parse_args()

# Guard nonsense combinations rather than letting them degrade into a silent
# no-op that looks like a legitimate run in the logs.
# Descriptions are per-domain and must stay paired: one branch on descriptions
# and the other on the hard-coded template would confound every comparison.
if (opts.desc_sketch is None) != (opts.desc_photo is None):
    missing = '--desc_photo' if opts.desc_photo is None else '--desc_sketch'
    given = '--desc_sketch' if opts.desc_photo is None else '--desc_photo'
    raise ValueError(
        f"{given} was given but {missing} was not. Class descriptions must be supplied for "
        "BOTH branches or neither: running one branch on VLM descriptions while the other "
        "keeps the hard-coded template changes two things at once, so the run would not be "
        f"comparable to any baseline. Pass {missing} as well, or drop {given}."
    )
# Every variant except the baseline needs the description files.
if opts.text_variant != 'template':
    missing = [f for f, v in (('--desc_sketch', opts.desc_sketch),
                              ('--desc_photo', opts.desc_photo)) if v is None]
    if missing:
        raise ValueError(
            f"--text_variant {opts.text_variant} needs class descriptions, but "
            f"{' and '.join(missing)} {'was' if len(missing) == 1 else 'were'} not given. "
            f"Pass both files, or use --text_variant template."
        )

if opts.aug_shared_encoder and opts.aug_identity_transform:
    if not opts.allow_degenerate_aug:
        raise ValueError(
            "--aug_shared_encoder and --aug_identity_transform are mutually exclusive by "
            "default. Run A removes clip_aug entirely, so an identity transform makes the "
            "main encoder see the same tensor twice and loss_aug collapses to InfoNCE(f, f). "
            "Run A and Run B are two separate runs -- pass exactly one of the two flags. "
            "If the collapsed term is the point of the experiment, pass "
            "--allow_degenerate_aug to opt in deliberately."
        )
    print("WARNING: degenerate aug mode -- photo_aug_feat is bit-identical to "
          "photo_feat. loss_aug reduces to InfoNCE(f, f): the positive term is "
          "saturated (cos=1) but the negative term remains active, acting as an "
          "intra-domain uniformity regularizer. This is intentional.")
if opts.aug_detach_view and not opts.aug_shared_encoder:
    print("[WARN] --aug_detach_view has no effect without --aug_shared_encoder: the clip_aug "
          "path does its own normalization and is left symmetric on purpose. Ignored.")
if opts.disable_aug_branch and (opts.aug_shared_encoder or opts.aug_identity_transform):
    flag = '--aug_shared_encoder' if opts.aug_shared_encoder else '--aug_identity_transform'
    raise ValueError(
        f"--disable_aug_branch cannot be combined with {flag}. --disable_aug_branch stops the "
        "dataset from emitting the augmented tensors at all, so loss_aug is skipped and "
        f"{flag} would be a silent no-op -- the run would be identical to a plain "
        "--disable_aug_branch run while its name/log claims otherwise."
    )
if opts.disable_aug_branch and opts.aug_side != 'both':
    raise ValueError(
        f"--disable_aug_branch cannot be combined with --aug_side {opts.aug_side}: there is no "
        "augmentation term left to take a side of, so the run would be a plain "
        "--disable_aug_branch run under a misleading name."
    )
if opts.prompt_branch != 'both' and opts.no_prompt_learning:
    raise ValueError(
        f"--prompt_branch {opts.prompt_branch} cannot be combined with --no_prompt_learning: the "
        "latter already removes every prompt (visual AND text), so the run would be a plain "
        "--no_prompt_learning run under a misleading name."
    )
