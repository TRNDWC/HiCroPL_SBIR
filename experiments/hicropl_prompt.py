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


def resolve_output_paths(run_opts):
    is_kaggle = os.path.exists('/kaggle/working')
    base_root = '/kaggle/working' if is_kaggle else os.getcwd()

    log_root = run_opts.log_root if run_opts.log_root else os.path.join(base_root, 'tb_logs')
    ckpt_root = run_opts.ckpt_root if run_opts.ckpt_root else os.path.join(base_root, 'saved_models')
    report_root = run_opts.report_root if run_opts.report_root else os.path.join(base_root, 'reports')

    exp_log_dir = os.path.join(log_root, run_opts.exp_name)
    exp_ckpt_dir = os.path.join(ckpt_root, run_opts.exp_name)
    exp_report_dir = os.path.join(report_root, run_opts.exp_name)

    os.makedirs(exp_log_dir, exist_ok=True)
    os.makedirs(exp_ckpt_dir, exist_ok=True)
    os.makedirs(exp_report_dir, exist_ok=True)

    return {
        'log_root': log_root,
        'ckpt_root': ckpt_root,
        'report_root': report_root,
        'exp_log_dir': exp_log_dir,
        'exp_ckpt_dir': exp_ckpt_dir,
        'exp_report_dir': exp_report_dir,
        'is_kaggle': is_kaggle,
    }

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

    g = torch.Generator()
    g.manual_seed(SEED)
    print(f"[CONFIG] Adapters ALWAYS enabled | adapter_reduction={opts.adapter_reduction}, image_adapter_m={opts.image_adapter_m}, text_adapter_m={opts.text_adapter_m}")

    # 1. Prepare Datasets
    output_paths = resolve_output_paths(opts)
    print(f"[PATHS] Kaggle mode: {output_paths['is_kaggle']}")
    print(f"[PATHS] TensorBoard root: {output_paths['log_root']}")
    print(f"[PATHS] Checkpoint root: {output_paths['ckpt_root']}")
    print(f"[PATHS] Report root: {output_paths['report_root']}")

    dataset_transforms = Sketchy.data_transform(opts)
    
    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    val_sketch = ValidDataset(opts, mode='sketch')
    val_photo = ValidDataset(opts, mode='photo')

    print(f"Train dataset: {len(train_dataset)} samples, {len(train_dataset.all_categories)} categories")
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
        worker_init_fn=seed_worker, generator=g,
    )
    val_sketch_loader = DataLoader(
        dataset=val_sketch, batch_size=opts.test_batch_size,
        num_workers=opts.workers, shuffle=False,
        worker_init_fn=seed_worker, generator=g,
    )
    val_photo_loader = DataLoader(
        dataset=val_photo, batch_size=opts.test_batch_size,
        num_workers=opts.workers, shuffle=False,
        worker_init_fn=seed_worker, generator=g,
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
    logger = TensorBoardLogger(output_paths['log_root'], name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_map_all' if opts.dataset != 'sketchy_ext' else 'val_map_200',
        dirpath=output_paths['exp_ckpt_dir'],
        filename="{epoch:02d}-{val_map_200:.4f}" if opts.dataset == 'sketchy_ext' else "{epoch:02d}-{val_map_all:.4f}",
        mode='max',
        save_last=True)

    ckpt_path = os.path.join(output_paths['exp_ckpt_dir'], 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    rich_progress_bar = RichProgressBar(
        leave=True
    )

    # 5. Initialize Trainer
    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        min_epochs=1, max_epochs=opts.epochs,
        benchmark=False,  # Set False for reproducibility (True causes CUDNN non-determinism)
        deterministic=True,
        logger=logger,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, rich_progress_bar]
    )

    # 6. Initialize Model
    if ckpt_path is None:
        custom_clip = CustomCLIP(opts, clip_model, clip_model_frozen)
        model = HiCroPL_SBIR(cfg=opts, args=opts, classnames=classnames, model=custom_clip)
    else:
        print ('resuming training from %s'%ckpt_path)
        # Note: Depending on Lightning version, PyTorch Lightning may require the architecture 
        # to be instantiated before load_from_checkpoint or handle it directly if args are passed correctly.
        custom_clip = CustomCLIP(opts, clip_model, clip_model_frozen)
        model = HiCroPL_SBIR.load_from_checkpoint(ckpt_path, cfg=opts, args=opts, classnames=classnames, model=custom_clip)

    print ('\nBeginning training HiCroPL-SBIR... Good luck!')
    trainer.fit(model, train_loader, [val_sketch_loader, val_photo_loader], ckpt_path=ckpt_path)

    print('\n[TRAINING COMPLETE]')
    print(f"[LOGDIR] {logger.log_dir}")
    print(f"[PLOT COMMAND] python read_output.py --logdir {logger.log_dir} --report_dir {output_paths['exp_report_dir']} --smooth 0.6")
