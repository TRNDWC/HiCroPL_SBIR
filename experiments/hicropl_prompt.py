import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from src.clip import clip
from src.model_hicropl import CustomCLIP, HiCroPL_SBIR
from src.dataset_retrieval import Sketchy, ValidDataset
from experiments.options import opts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    SEED = 42
    # Set seed for reproducibility — bao gồm Python, NumPy, PyTorch, CUDA
    pl.seed_everything(SEED, workers=True)

    # Force a single backbone across all branches for stable comparisons.
    if opts.backbone != 'ViT-B/32':
        print(f"[WARN] Overriding backbone {opts.backbone} -> ViT-B/32")
    opts.backbone = 'ViT-B/32'

    # --no_prompt_learning ablation: force a vanilla (non-prompted) CLIP build
    # -- src/clip/model.py only attaches the HiCroPL prompt-injection blocks
    # when design_details['trainer'] == 'HiCroPL'/'MaPLe'; any other value
    # (e.g. 'CoOp') builds the plain VisionTransformer/ResidualAttentionBlock,
    # so encode_image/encode_text work standalone with no prompt tensors at
    # all. Must happen BEFORE load_clip_to_cpu(opts) below.
    if opts.no_prompt_learning:
        if opts.clip_trainer != 'CoOp':
            print(f"[CONFIG] --no_prompt_learning: overriding clip_trainer {opts.clip_trainer} -> CoOp (vanilla CLIP, no prompt injection)")
        opts.clip_trainer = 'CoOp'

    def seed_worker(worker_id):
        """Seed numpy và random trong mỗi DataLoader worker (fix np.random.choice non-determinism)."""
        worker_seed = torch.initial_seed() % 2**32
        import numpy as np
        import random
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        return torch.utils.data.default_collate(batch)

    g = torch.Generator()
    g.manual_seed(SEED)
    print(f"[CONFIG] Running HiCroPL with backbone {opts.backbone}")

    # 1. Prepare Datasets
    if opts.eval_mode == 'fine_grained':
        print(f"[CONFIG] Loading data in fine-grained mode")
        from src_fg.dataset_fg import SketchyDataset as SketchyDatasetFG
        train_dataset = SketchyDatasetFG(opts, mode='train')
        val_dataset = SketchyDatasetFG(opts, mode='test')
    else:
        dataset_transforms = Sketchy.data_transform(opts)
        # Augmentation branch is ON by default; --disable_aug_branch removes it
        # entirely (no aug tensors emitted, no clip_aug built, no aug loss).
        if opts.disable_aug_branch:
            aug_photo = aug_sketch = None
            print("[ABLATION] --disable_aug_branch: augmentation branch fully removed.")
        else:
            # Run B routes both of these to Sketchy.data_transform internally, so
            # the branch below is unchanged -- the flag is read inside the two
            # static methods (src/dataset_retrieval.py).
            aug_photo = Sketchy.data_transform_aug_photo(opts)
            aug_sketch = Sketchy.data_transform_aug_sketch(opts)
            if opts.aug_identity_transform:
                print("[ABLATION] --aug_identity_transform (Run B): aug transforms replaced by "
                      "Sketchy.data_transform for BOTH photo and sketch -- clip_aug still built, "
                      "aug views must be bit-wise equal to the clean ones (checked on batch 0).")
            if opts.aug_shared_encoder:
                print("[ABLATION] --aug_shared_encoder (Run A): clip_aug NOT built; the augmented "
                      "views go through the main encoder with the same prompts. "
                      f"aug_detach_view={int(opts.aug_detach_view)}.")
        if opts.cross_dataset_eval:
            if not opts.eval_dataset or not opts.eval_data_dir:
                raise ValueError("--cross_dataset_eval requires both --eval_dataset and --eval_data_dir.")
            if opts.gzs_eval:
                raise ValueError("--cross_dataset_eval and --gzs_eval are mutually exclusive "
                                  "(GZS mixes SEEN classes of --dataset; cross-dataset eval classes "
                                  "are all unseen by construction).")
            if opts.eval_mode_gzs:
                raise ValueError("--cross_dataset_eval and --eval_mode_gzs are mutually exclusive.")
            print(f"[CONFIG] --cross_dataset_eval: training on ALL categories of '{opts.dataset}', "
                  f"evaluating on '{opts.eval_dataset}' ({opts.eval_data_dir})")
        if opts.eval_mode_gzs and opts.gzs_eval:
            raise ValueError("--eval_mode_gzs and --gzs_eval are mutually exclusive.")
        if (opts.eval_mode_gzs_ocean or opts.eval_mode_gzs_drclip) and (
                opts.eval_mode_gzs or opts.gzs_eval or opts.cross_dataset_eval):
            raise ValueError("--eval_mode_gzs_ocean/--eval_mode_gzs_drclip are mutually exclusive with "
                              "--eval_mode_gzs, --gzs_eval, and --cross_dataset_eval.")
        if opts.eval_mode_gzs_ocean and opts.eval_mode_gzs_drclip:
            raise ValueError("--eval_mode_gzs_ocean and --eval_mode_gzs_drclip are mutually exclusive.")
        if opts.eval_mode_gzs:
            print(f"[CONFIG] --eval_mode_gzs: gallery = P^s (ALL train photos of every seen "
                  f"class of '{opts.dataset}') union P^u_test (unseen-class photos, unchanged); "
                  f"query stays S^u_test (unchanged).")
        if opts.eval_mode_gzs_ocean or opts.eval_mode_gzs_drclip:
            _proto, _basis = (('OCEAN (Zhu et al., ICME 2020)', '|C^u|') if opts.eval_mode_gzs_ocean
                              else ('Dr. CLIP (Li et al., ACM MM 2024)', '|C^s|'))
            print(f"[CONFIG] {_proto} GZS-SBIR protocol -- C^g = C^u union "
                  f"round(0.2*{_basis}) randomly chosen whole seen classes of '{opts.dataset}'; "
                  f"both query and gallery are drawn from C^g (seen-class sketches ARE queried, "
                  f"not just distractors).")
        train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False,
                                transform_aug_photo=aug_photo, transform_aug_sketch=aug_sketch)
        print(f"[CONFIG] Loading validation data in category mode")
        val_sketch = ValidDataset(opts, mode='sketch')
        val_photo = ValidDataset(opts, mode='photo')
        if opts.eval_mode_gzs:
            print(f"GZS_FP | on=1 | n_gallery_seen={val_photo.n_gallery_seen} | "
                  f"n_gallery_unseen={val_photo.n_gallery_unseen} | "
                  f"n_gallery_total={val_photo.n_gallery_seen + val_photo.n_gallery_unseen} | "
                  f"n_query={val_sketch.n_query}")
        if opts.eval_mode_gzs_ocean or opts.eval_mode_gzs_drclip:
            _tag = 'OCEAN' if opts.eval_mode_gzs_ocean else 'Dr.CLIP'
            print(f"GZS_{_tag}_FP | on=1 | n_test_classes_total={len(val_photo.all_categories)} | "
                  f"n_gallery_seen={val_photo.n_gallery_seen} | "
                  f"n_gallery_unseen={val_photo.n_gallery_unseen} | "
                  f"n_gallery_total={val_photo.n_gallery_seen + val_photo.n_gallery_unseen} | "
                  f"n_query_seen={val_sketch.n_query_seen} | n_query_unseen={val_sketch.n_query_unseen} | "
                  f"n_query_total={val_sketch.n_query}")

    print(f"Train dataset: {len(train_dataset)} samples, {len(train_dataset.all_categories)} categories")
    if opts.eval_mode == 'fine_grained':
        print(f"Val dataset: {len(val_dataset)} samples")
        print(f"[DEBUG] Val categories (first 5): {val_dataset.all_categories[:5]}")
    else:
        print(f"Val sketch dataset: {len(val_sketch)} samples")
        print(f"Val photo dataset: {len(val_photo)} samples")
        # Debug: verify category ordering is consistent across runs
        print(f"[DEBUG] Val categories (first 5): {val_sketch.all_categories[:5]}")
        print(f"[DEBUG] Val photo categories (first 5): {val_photo.all_categories[:5]}")
        assert val_sketch.all_categories == val_photo.all_categories, \
            "CRITICAL: sketch and photo category lists differ! Fix dataset loading."

    # 2. Prepare DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size,
        num_workers=opts.workers, shuffle=True,
        worker_init_fn=seed_worker, generator=g, collate_fn=collate_fn,
    )
    if opts.eval_mode == 'fine_grained':
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=opts.test_batch_size,
            num_workers=opts.workers, shuffle=False,
            worker_init_fn=seed_worker, generator=g, collate_fn=collate_fn,
        )
    else:
        val_sketch_loader = DataLoader(
            dataset=val_sketch, batch_size=opts.test_batch_size,
            num_workers=opts.workers, shuffle=False,
            worker_init_fn=seed_worker, generator=g, collate_fn=collate_fn,
        )
        val_photo_loader = DataLoader(
            dataset=val_photo, batch_size=opts.test_batch_size,
            num_workers=opts.workers, shuffle=False,
            worker_init_fn=seed_worker, generator=g, collate_fn=collate_fn,
        )

    # 3. Setup CLIP backbones
    from src.utils import load_clip_to_cpu
    print("Loading CLIP models...")

    clip_model = load_clip_to_cpu(opts).to(device)
    clip_model.float() # Training prompt in fp32

    # Extract classnames for Context Learner initialization
    classnames = list(train_dataset.all_categories)

    # Sample real photo images once, used only for k-means-based photo prompt
    # layer-0 initialization (one-time, see VisualVisualPromptLearner) --
    # no persistent regularization loss pulls the prompt back toward this
    # afterward. Photo-only by design -- cheap, done once before training starts.
    # Stratified by category: plain uniform sampling over 57587 images / 104
    # categories left ~30 categories with zero representation in expectation
    # (coupon-collector effect at n=128), so we guarantee >=1 image/category
    # instead of leaving coverage to chance.
    photo_idx = 0 if opts.eval_mode == 'fine_grained' else 1
    sketch_idx = 1 if opts.eval_mode == 'fine_grained' else 0
    target_total = min(128, len(train_dataset))
    min_per_category = opts.kmeans_min_per_category

    category_indices = {}
    for idx, sk_path in enumerate(train_dataset.all_sketches_path):
        category = sk_path.split(os.path.sep)[-2]
        category_indices.setdefault(category, []).append(idx)

    categories = list(category_indices.keys())
    per_category = max(min_per_category, target_total // len(categories))

    sample_photo_list = []
    sample_sketch_list = []
    for category in categories:
        indices = category_indices[category]
        chosen = random.sample(indices, min(per_category, len(indices)))
        for idx in chosen:
            item = train_dataset[idx]
            if item is None:
                continue
            sample_photo_list.append(item[photo_idx])
            sample_sketch_list.append(item[sketch_idx])
    sample_photo_images = torch.stack(sample_photo_list) if sample_photo_list else None
    sample_sketch_images = torch.stack(sample_sketch_list) if sample_sketch_list else None
    print(f"[CONFIG] Sampled {len(sample_photo_list)} real photo + {len(sample_sketch_list)} real sketch images "
          f"for k-means prompt init (stratified, {per_category}/category across {len(categories)} categories)")

    # 4. Setup Checkpointing and Logger
    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    # Log hyperparameters to TensorBoard's HParams and Text tabs
    opts_dict = vars(opts)
    logger.log_hyperparams(opts_dict)

    # Generate a Markdown table for TensorBoard's Text tab
    md_table = "### Training Hyperparameters\n\n| Parameter | Value |\n|---|---|\n"
    for key, value in sorted(opts_dict.items()):
        md_table += f"| **{key}** | `{value}` |\n"
    logger.experiment.add_text("hyperparameters", md_table, global_step=0)

    if opts.eval_mode == 'fine_grained':
        checkpoint_monitor = 'top1'
        checkpoint_filename = '{epoch:02d}-{top1:.4f}'
    else:
        # Must mirror _on_validation_epoch_end_category's map_k selection exactly
        # (src/model_hicropl.py) -- that's what decides whether 'val_map_200' or
        # 'val_map_all' actually gets logged. sketchy_2 and sketchy_ext both hit
        # the map_k=200 branch there. --cross_dataset_eval always logs mAP@all
        # regardless of --dataset, so it must not fall into the val_map_200 branch.
        use_map_200 = opts.dataset in ('sketchy_2', 'sketchy_ext') and not opts.cross_dataset_eval
        checkpoint_monitor = 'val_map_200' if use_map_200 else 'val_map_all'
        checkpoint_filename = '{epoch:02d}-{val_map_200:.4f}' if use_map_200 else '{epoch:02d}-{val_map_all:.4f}'

    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_monitor,
        dirpath='saved_models/%s' % opts.exp_name,
        filename=checkpoint_filename,
        mode='max',
        save_last=False)

    ckpt_path = os.path.join('saved_models/%s'%opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    # 5. Initialize Trainer
    rich_progress_bar = RichProgressBar(
        leave=True
    )

    callbacks = [checkpoint_callback, rich_progress_bar]
    # Opt-in-only diagnostic: gradient/param-norm CSV logger (tools/diag_gradlog.py).
    # No effect unless HICROPL_GRADLOG_CSV is set -- does not touch loss, optimizer,
    # seed, lr, or any other hyperparameter; read-only, one Lightning hook.
    gradlog_csv = os.environ.get('HICROPL_GRADLOG_CSV')
    if gradlog_csv:
        from tools.diag_gradlog import GradLogCallback
        callbacks.append(GradLogCallback(gradlog_csv))
        print(f"[CONFIG] HICROPL_GRADLOG_CSV set: logging grad/param norms to {gradlog_csv}")

    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        min_epochs=1, max_epochs=opts.epochs,
        benchmark=False,  # Set False for reproducibility (True causes CUDNN non-determinism)
        deterministic=True,
        logger=logger,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        callbacks=callbacks
    )

    # 6. Initialize Model
    if ckpt_path is None:
        custom_clip = CustomCLIP(opts, clip_model, classnames=classnames, sample_photo_images=sample_photo_images, sample_sketch_images=sample_sketch_images)
        if opts.eval_mode == 'fine_grained':
            from src_fg.model_hicropl_fg import HiCroPL_SBIR_FG
            model = HiCroPL_SBIR_FG(cfg=opts, args=opts, classnames=classnames, model=custom_clip)
        else:
            model = HiCroPL_SBIR(cfg=opts, args=opts, classnames=classnames, model=custom_clip)
    else:
        print ('resuming training from %s'%ckpt_path)
        # Note: Depending on Lightning version, PyTorch Lightning may require the architecture 
        # to be instantiated before load_from_checkpoint or handle it directly if args are passed correctly.
        custom_clip = CustomCLIP(opts, clip_model, classnames=classnames, sample_photo_images=sample_photo_images, sample_sketch_images=sample_sketch_images)
        if opts.eval_mode == 'fine_grained':
            from src_fg.model_hicropl_fg import HiCroPL_SBIR_FG
            model = HiCroPL_SBIR_FG.load_from_checkpoint(ckpt_path, cfg=opts, args=opts, classnames=classnames, model=custom_clip)
        else:
            model = HiCroPL_SBIR.load_from_checkpoint(ckpt_path, cfg=opts, args=opts, classnames=classnames, model=custom_clip)

    print ('\nBeginning training HiCroPL-SBIR... Good luck!')
    if opts.eval_mode == 'fine_grained':
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_loader, [val_sketch_loader, val_photo_loader], ckpt_path=ckpt_path)
