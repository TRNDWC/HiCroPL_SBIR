import torch
import torch.nn as nn
import torch.nn.functional as F
from src_fg.jigsaw import JigsawNet
from collections import defaultdict

from src.model_hicropl import HiCroPL_SBIR
from src.losses_hicropl import loss_fn_hicropl, cross_loss
from src_fg.dataset_fg import permute_patch, generate_perm


class HiCroPL_SBIR_FG(HiCroPL_SBIR):
    def __init__(self, cfg, args, classnames, model):
        super().__init__(cfg, args, classnames, model)
        self.eval_mode = 'fine_grained'
        self.val = defaultdict(lambda: {
            'val_img_features': [],
            'val_img_names': [],
            'val_sk_features': [],
            'val_sk_names': [],
        })
        self.train_loss_epoch = []
        self.patch_loss_epoch = []
        self.cjs_loss_epoch = []
        self.jigsaw_net = None

    def training_step(self, batch, batch_idx):
        # batch format from SketchyDatasetFG (train): img, sk, img_aug, sk_aug, neg, label
        img_tensor = batch[0]
        sk_tensor = batch[1]
        neg_tensor = batch[4]
        reordered = (batch[1], batch[0], batch[4], batch[3], batch[2], batch[5])

        features = self.model(reordered, self.classnames)
        loss = loss_fn_hicropl(self.args, features)

        # --- Patch Shuffle Loss (always on for FG) ---
        num_split = getattr(self.args, 'patch_split', 2)
        B = img_tensor.size(0)
        img_shuffle_list = []
        sk_shuffle_list = []
        perm_list = []
        for i in range(B):
            perm = generate_perm(num_split=num_split)
            perm_list.append(perm.to(img_tensor.device))
            img_shuffle = permute_patch(img_tensor[i].detach().cpu(), perm, num_split=num_split).to(img_tensor.device)
            sk_shuffle = permute_patch(sk_tensor[i].detach().cpu(), perm, num_split=num_split).to(sk_tensor.device)
            img_shuffle_list.append(img_shuffle)
            sk_shuffle_list.append(sk_shuffle)

        img_shuffle_batch = torch.stack(img_shuffle_list)
        sk_shuffle_batch = torch.stack(sk_shuffle_list)

        img_shuffle_feat = self.extract_eval_features(img_shuffle_batch, modality='photo')
        sk_shuffle_feat = self.extract_eval_features(sk_shuffle_batch, modality='sketch')

        temperature = getattr(self.args, 'temperature', 0.07)
        patch_loss = cross_loss(sk_shuffle_feat, img_shuffle_feat, temperature)
        lambda_patch = 1.0

        total_loss = loss + lambda_patch * patch_loss

        self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('patch_loss', patch_loss, on_step=False, on_epoch=True, prog_bar=False)

        self.train_loss_epoch.append(total_loss.detach())
        self.patch_loss_epoch.append(patch_loss.detach())

        # --- Conditional Cross-modal Jigsaw Loss (L_cjs) ---
        # Build per-patch features for sketch and shuffled-sketch, predict original locations
        num_patches = num_split ** 2

        def extract_patch_feats(images):
            # images: (B, C, H, W)
            B_img, C, H, W = images.shape
            patch_size = H // num_split
            patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            # patches: B, C, num_split, num_split, ph, pw
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            patches = patches.view(B_img, num_patches, C, patch_size, patch_size)
            patches_flat = patches.view(-1, C, patch_size, patch_size)
            patches_resized = F.interpolate(patches_flat, size=(224, 224), mode='bilinear', align_corners=False)
            out = self.model.extractor_sketch(patches_resized)
            feats = out["image_features"]
            feats = feats.view(B_img, num_patches, -1)
            return feats

        # Extract patch features (original & shuffled)
        patches_feat_s = extract_patch_feats(sk_tensor)
        patches_feat_s_prime = extract_patch_feats(sk_shuffle_batch)

        # Global context features (prompted + distill mix)
        z_s = self.extract_eval_features(sk_tensor, modality='sketch')
        z_p_plus = self.extract_eval_features(img_tensor, modality='photo')
        z_p_minus = self.extract_eval_features(neg_tensor, modality='photo')

        # Lazy init of jigsaw_net
        if self.jigsaw_net is None:
            feat_dim = patches_feat_s.shape[-1]
            self.jigsaw_net = JigsawNet(num_patches=num_patches, feat_dim=feat_dim).to(self.device)

        # Predict logits: shape (B, P, P)
        logits_r = self.jigsaw_net(patches_feat_s_prime, context=z_s)
        logits_r_plus = self.jigsaw_net(patches_feat_s_prime, context=z_p_plus)
        logits_r_minus = self.jigsaw_net(patches_feat_s_prime, context=z_p_minus)

        # Prepare labels from perm_list: each perm gives source index for each target position
        labels = torch.stack(perm_list, dim=0)  # (B, P)
        labels = labels.long().to(self.device)

        ce = nn.CrossEntropyLoss()
        Bp = B * num_patches
        loss_r = ce(logits_r.view(Bp, num_patches), labels.view(Bp))
        loss_r_plus = ce(logits_r_plus.view(Bp, num_patches), labels.view(Bp))
        loss_r_minus = ce(logits_r_minus.view(Bp, num_patches), labels.view(Bp))

        margin = F.relu(loss_r_plus - loss_r_minus)
        cjs_loss = loss_r + margin
        lambda_cjs = getattr(self.args, 'lambda_cjs', 1.0)

        total_loss = total_loss + lambda_cjs * cjs_loss

        self.log('cjs_loss', cjs_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.cjs_loss_epoch.append(cjs_loss.detach())

        # update logged train loss (includes cjs now)
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=False)

        return total_loss

    def on_train_epoch_end(self):
        if len(self.train_loss_epoch) == 0:
            return

        epoch_train_loss = torch.stack(self.train_loss_epoch).mean()
        epoch_patch_loss = torch.stack(self.patch_loss_epoch).mean()
        epoch_cjs_loss = torch.stack(self.cjs_loss_epoch).mean() if len(self.cjs_loss_epoch) > 0 else torch.tensor(0.0, device=self.device)
        epoch_base_loss = epoch_train_loss - epoch_patch_loss

        self.print(
            f"loss: {epoch_base_loss.item():.4f}, patch_loss: {epoch_patch_loss.item():.4f}, "
            f"cjs_loss: {epoch_cjs_loss.item():.4f}, train_loss: {epoch_train_loss.item():.4f}"
        )

        self.train_loss_epoch.clear()
        self.patch_loss_epoch.clear()
        self.cjs_loss_epoch.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        sk_tensor, sk_name, img_tensor, pos_name, label = batch

        sk_feature = self.extract_eval_features(sk_tensor, modality='sketch')
        img_feature = self.extract_eval_features(img_tensor, modality='photo')

        if torch.is_tensor(label):
            label_list = label.detach().cpu().tolist()
        else:
            label_list = list(label)

        for i in range(len(label_list)):
            lab = label_list[i]
            self.val[lab]['val_sk_features'].append(sk_feature[i].detach().cpu())
            self.val[lab]['val_sk_names'].append(sk_name[i])

            p_name = pos_name[i]
            if p_name not in self.val[lab]['val_img_names']:
                self.val[lab]['val_img_names'].append(p_name)
                self.val[lab]['val_img_features'].append(img_feature[i].detach().cpu())

    def on_validation_epoch_end(self):
        if len(self.val) == 0:
            self.print("Warning: No fine-grained data collected. Skipping FG metrics.")
            return

        top1_list, top5_list = [], []

        for category, bucket in self.val.items():
            rank = torch.zeros(len(bucket['val_sk_names']), device=self.device)

            if len(bucket['val_img_features']) == 0:
                continue

            val_img_feature = torch.stack(bucket['val_img_features'])

            for num, sketch_feature in enumerate(bucket['val_sk_features']):
                s_name = bucket['val_sk_names'][num]
                sk_query_name = s_name.split('/')[-1].split('-')[:-1][0]
                position_query = bucket['val_img_names'].index(sk_query_name)

                distance = self.distance_fn(sketch_feature.unsqueeze(0), val_img_feature)
                target_distance = self.distance_fn(
                    sketch_feature.unsqueeze(0),
                    val_img_feature[position_query].unsqueeze(0)
                )
                rank[num] = distance.le(target_distance).sum()

            top1_list.append(rank.le(1).float().mean().item())
            top5_list.append(rank.le(5).float().mean().item())

        if len(top1_list) == 0:
            self.print("Warning: No valid categories for FG evaluation.")
            self.val.clear()
            return

        top1 = sum(top1_list) / len(top1_list)
        top5 = sum(top5_list) / len(top5_list)

        self.log('top1', top1, on_step=False, on_epoch=True, prog_bar=False)
        self.log('top5', top5, on_step=False, on_epoch=True, prog_bar=False)

        if self.global_step > 0:
            self.best_metric = max(self.best_metric, top1)
        self.log('best_fg_acc@1', self.best_metric, on_epoch=True, prog_bar=False)

        self.print(f'top1: {top1:.4f}, top5: {top5:.4f}, Best: {self.best_metric:.4f}')
        self.val.clear()
