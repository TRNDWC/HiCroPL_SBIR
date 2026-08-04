import torch
from collections import defaultdict

from src.model_hicropl import HiCroPL_SBIR
from src.losses_hicropl import loss_fn_hicropl


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

    def training_step(self, batch, batch_idx):
        # batch format from SketchyDatasetFG (train): img, sk, img_aug, sk_aug, neg, label
        img_tensor = batch[0]
        sk_tensor = batch[1]
        neg_tensor = batch[4]
        reordered = (batch[1], batch[0], batch[4], batch[3], batch[2], batch[5])

        features = self.model(reordered, self.classnames)
        # loss_fn_hicropl trả (total_loss, loss_dict); gán thẳng vào 1 biến sẽ
        # đưa nguyên tuple vào self.log và làm hỏng bước train.
        total_loss, loss_dict = loss_fn_hicropl(self.args, features)

        self.log('loss', total_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=False)
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor) or v > 0:
                self.log(k, v, on_step=False, on_epoch=True, prog_bar=False)

        self.train_loss_epoch.append(total_loss.detach())

        return total_loss

    def on_train_epoch_end(self):
        if len(self.train_loss_epoch) == 0:
            return

        epoch_train_loss = torch.stack(self.train_loss_epoch).mean()
        self.print(f"train_loss: {epoch_train_loss.item():.4f}")
        self.train_loss_epoch.clear()

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
