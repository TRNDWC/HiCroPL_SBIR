import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision, retrieval_precision
import pytorch_lightning as pl

from src.clip import clip
from experiments.options import opts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freeze_model(m):
    m.requires_grad_(False)

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

def set_ln_to_train(m):
    """Set LayerNorm modules to train mode while keeping others in eval"""
    if isinstance(m, torch.nn.LayerNorm):
        m.train()
    
class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts
        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        self.clip.apply(freeze_all_but_bn)
        
        # Set CLIP to eval mode but LayerNorms to train mode
        self.clip.eval()
        self.clip.apply(set_ln_to_train)

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance = lambda x, y: F.cosine_similarity(x, y)
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.3)
        self.val_step_outputs_sk = []
        self.val_step_outputs_ph = []
        self.best_metric = 0.0
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.clip.parameters(), 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt] + [self.img_prompt], 'lr': self.opts.prompt_lr}])
        return optimizer

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.clip.encode_image(
                data, self.img_prompt.expand(data.shape[0], -1, -1))
        else:
            feat = self.clip.encode_image(
                data, self.sk_prompt.expand(data.shape[0], -1, -1))
        return feat

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        img_tensor, category = batch
        if dataloader_idx == 0:
            sk_feat = self.forward(img_tensor, dtype='sketch')
            self.val_step_outputs_sk.append((sk_feat, category))
        else:
            img_feat = self.forward(img_tensor, dtype='image')
            self.val_step_outputs_ph.append((img_feat, category))

    def on_validation_epoch_end(self):
        query_len = len(self.val_step_outputs_sk)
        gallery_len = len(self.val_step_outputs_ph)
        
        if query_len == 0 or gallery_len == 0:
            return
        
        query_feat_all = torch.cat([self.val_step_outputs_sk[i][0] for i in range(query_len)])
        gallery_feat_all = torch.cat([self.val_step_outputs_ph[i][0] for i in range(gallery_len)])
        
        all_sketch_category = np.array(sum([list(self.val_step_outputs_sk[i][1].detach().cpu().numpy()) for i in range(query_len)], []))
        all_photo_category = np.array(sum([list(self.val_step_outputs_ph[i][1].detach().cpu().numpy()) for i in range(gallery_len)], []))
        
        ## mAP category-level SBIR Metrics
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        precision = torch.zeros(len(query_feat_all))
        
        # Set metrics according to dataset (following paper)
        if self.opts.dataset == 'sketchy_ext':
            map_k = 200  # mAP@200
            p_k = 200    # P@200
        elif self.opts.dataset == 'tuberlin':
            map_k = 0    # mAP@all
            p_k = 100    # P@100
        elif self.opts.dataset == 'quickdraw':
            map_k = 0    # mAP@all
            p_k = 200    # P@200
        else:  # sketchy (basic)
            map_k = 0    # mAP@all
            p_k = 200    # P@200
                
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_sketch_category[idx]
            distance = self.distance(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool, device=device)
            target[np.where(all_photo_category == category)] = True
            
            if map_k != 0:
                top_k_actual = min(map_k, len(gallery)) 
                ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu(), top_k=top_k_actual)
            else: 
                ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())
                
            precision[idx] = retrieval_precision(distance.cpu(), target.cpu(), top_k=p_k)
            
        mAP = torch.mean(ap)
        precision = torch.mean(precision)
        self.log('mAP', mAP, on_step=False, on_epoch=True)
        if self.global_step > 0:
            self.best_metric = self.best_metric if (self.best_metric > mAP.item()) else mAP.item()
        
        if map_k != 0:
            print('mAP@{}: {}, P@{}: {}, Best mAP: {}'.format(map_k, mAP.item(), p_k, precision, self.best_metric))
        else:
            print('mAP@all: {}, P@{}: {}, Best mAP: {}'.format(mAP.item(), p_k, precision, self.best_metric))
        
        # Clear for next epoch
        self.val_step_outputs_sk.clear()
        self.val_step_outputs_ph.clear()