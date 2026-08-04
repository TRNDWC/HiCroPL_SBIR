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
from src.run_logging import RunCSVLogger, make_run_dir, setup_run_logger
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
        train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
        print(f"[CONFIG] Loading validation data in category mode")
        val_sketch = ValidDataset(opts, mode='sketch')
        val_photo = ValidDataset(opts, mode='photo')

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
    from src.utils import load_clip_to_cpu, load_clip_to_cpu_teacher
    print("Loading CLIP models...")
    
    clip_model = load_clip_to_cpu(opts).to(device)
    clip_model.float() # Training prompt in fp32
    
    clip_model_frozen = load_clip_to_cpu_teacher(opts).to(device)
    clip_model_frozen.float()
    clip_model_frozen.eval()
    
    # Extract classnames for Context Learner initialization
    classnames = list(train_dataset.all_categories)

    # 4. Setup Checkpointing and Logger
    logger = TensorBoardLogger(opts.log_dir, name=opts.exp_name)

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
        checkpoint_monitor = 'val_map_200' if opts.dataset == 'sketchy_ext' else 'val_map_all'
        checkpoint_filename = '{epoch:02d}-{val_map_200:.4f}' if opts.dataset == 'sketchy_ext' else '{epoch:02d}-{val_map_all:.4f}'

    ckpt_dir = os.path.join(opts.save_dir, opts.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Mỗi lần chạy một thư mục riêng -> chạy lại cùng exp_name không đè log cũ
    run_dir, run_id = make_run_dir(opts.log_dir, opts.exp_name, opts.run_id or None)
    opts.run_id = run_id

    run_logger = setup_run_logger(run_dir, opts.exp_name)
    run_logger.info('Checkpoint dir : %s', os.path.abspath(ckpt_dir))
    run_logger.info('TensorBoard dir: %s', os.path.abspath(os.path.join(opts.log_dir, opts.exp_name)))
    run_logger.info('Run dir        : %s', os.path.abspath(run_dir))
    run_logger.info('Summary CSV    : %s', os.path.abspath(opts.summary_csv))
    run_logger.info('Train %d samples, %d categories | val sketch/photo: %s',
                    len(train_dataset), len(train_dataset.all_categories),
                    len(val_dataset) if opts.eval_mode == 'fine_grained'
                    else f'{len(val_sketch)}/{len(val_photo)}')

    csv_logger = RunCSVLogger(
        cfg=opts,
        run_dir=run_dir,
        exp_name=opts.exp_name,
        summary_csv=opts.summary_csv,
        run_id=run_id,
        logger=run_logger,
        log_every_n_steps=opts.log_every_n_steps,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_monitor,
        dirpath=ckpt_dir,
        filename=checkpoint_filename,
        mode='max',
        save_top_k=opts.save_top_k,
        save_last=opts.save_last)

    ckpt_path = os.path.join(ckpt_dir, 'last.ckpt')
    if opts.no_resume or not os.path.exists(ckpt_path):
        if opts.no_resume and os.path.exists(ckpt_path):
            print(f'[CONFIG] --no_resume: bỏ qua {ckpt_path}, train từ đầu')
        elif not opts.save_last:
            # Auto-resume dựa vào last.ckpt, mà last.ckpt chỉ được ghi khi save_last=True.
            print('[CONFIG] --save_last chưa bật -> sẽ không có last.ckpt để auto-resume')
        ckpt_path = None
    else:
        print('resuming training from %s' % ckpt_path)

    # 5. Initialize Trainer
    rich_progress_bar = RichProgressBar(
        leave=True
    )

    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        min_epochs=1, max_epochs=opts.epochs,
        benchmark=False,  # Set False for reproducibility (True causes CUDNN non-determinism)
        deterministic=True,
        logger=logger,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, rich_progress_bar, csv_logger]
    )

    # 6. Initialize Model
    if ckpt_path is None:
        custom_clip = CustomCLIP(opts, clip_model, clip_model_frozen, classnames=classnames)
        if opts.eval_mode == 'fine_grained':
            from src_fg.model_hicropl_fg import HiCroPL_SBIR_FG
            model = HiCroPL_SBIR_FG(cfg=opts, args=opts, classnames=classnames, model=custom_clip)
        else:
            model = HiCroPL_SBIR(cfg=opts, args=opts, classnames=classnames, model=custom_clip)
    else:
        # Note: Depending on Lightning version, PyTorch Lightning may require the architecture
        # to be instantiated before load_from_checkpoint or handle it directly if args are passed correctly.
        custom_clip = CustomCLIP(opts, clip_model, clip_model_frozen, classnames=classnames)
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
