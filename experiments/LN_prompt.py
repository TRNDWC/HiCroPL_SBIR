import os
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model_LN_prompt import Model
from src.dataset_retrieval import Sketchy, ValidDataset
from experiments.options import opts

if __name__ == '__main__':
    dataset_transforms = Sketchy.data_transform(opts)

    from src.dataset_retrieval import ValidDataset
    
    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    val_sketch = ValidDataset(opts, mode='sketch')
    val_photo = ValidDataset(opts, mode='photo')

    print(f"Train dataset: {len(train_dataset)} samples, {len(train_dataset.all_categories)} categories")
    print(f"Val sketch dataset: {len(val_sketch)} samples")
    print(f"Val photo dataset: {len(val_photo)} samples")

    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    val_sketch_loader = DataLoader(dataset=val_sketch, batch_size=opts.test_batch_size, num_workers=opts.workers, shuffle=False)
    val_photo_loader = DataLoader(dataset=val_photo, batch_size=opts.test_batch_size, num_workers=opts.workers, shuffle=False)

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='mAP',
        dirpath='saved_models/%s'%opts.exp_name,
        filename="{epoch:02d}-{mAP:.2f}",
        mode='max',
        save_last=True)

    ckpt_path = os.path.join('saved_models/%s'%opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    trainer = Trainer(accelerator="gpu", devices=1,
        min_epochs=1, max_epochs=opts.epochs,
        benchmark=True,
        logger=logger,
        # val_check_interval=10, 
        # accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        # resume_from_checkpoint=ckpt_path,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback]
    )

    if ckpt_path is None:
        model = Model()
        # for name, p in model.named_parameters():
        #     if p.requires_grad == True:
        #         print(name)
    else:
        print ('resuming training from %s'%ckpt_path)
        model = Model().load_from_checkpoint(ckpt_path)

    print ('beginning training...good luck...')
    trainer.fit(model, train_loader, [val_sketch_loader, val_photo_loader])