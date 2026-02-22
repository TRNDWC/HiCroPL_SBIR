import os
import glob
import numpy as np
import torch
import math 
from torch.nn import functional as F

from torchvision import transforms
from PIL import Image, ImageOps

UNSEEN_CLASSES=[
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",]

def generate_perm(num_split = 2):
    return torch.randperm(num_split**2)

def permute_patch(images, perm, num_split = 2):
    image_size = 224
    patch_size = math.floor(image_size/num_split)
    interpolated_size = (patch_size*num_split, patch_size*num_split)
    perm_inds = []
    for a in range(num_split):
        for b in range(num_split):
            perm_inds.append([a, b])
            
    images = F.interpolate(images.unsqueeze(0), interpolated_size).squeeze(0)

    image_shuffle = torch.zeros(images.size()).type_as(images.data)
    for i_num, i_perm in enumerate(perm):
        x_source, y_source =  perm_inds[i_perm]
        x_target, y_target = perm_inds[i_num]
        image_shuffle[:, x_target * patch_size: (x_target + 1) * patch_size, y_target * patch_size: (y_target + 1) * patch_size] \
                = images[:, x_source * patch_size: (x_source + 1) * patch_size,
                                                        y_source * patch_size: (y_source + 1) * patch_size]

    return image_shuffle
        
def aumented_transform():
    transform_list = [
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.8),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomRotation(70),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(
            0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


def normal_transform():
    dataset_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    return dataset_transforms


class SketchyDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        unseen_classes = UNSEEN_CLASSES["sketchy_2"]

        self.all_categories = os.listdir(
            os.path.join(self.args.root, 'sketch'))
        self.transform = normal_transform()
        self.aumentation = aumented_transform()

        if self.mode == "train":
            self.all_categories = list(
                set(self.all_categories) - set(unseen_classes))
        else:
            self.all_categories = list(set(unseen_classes))

        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            self.all_sketches_path.extend(
                glob.glob(os.path.join(self.args.root, 'sketch', category, '*')))
            self.all_photos_path[category] = glob.glob(
                os.path.join(self.args.root, 'photo', category, 'n*'))

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, index):
        sk_path = self.all_sketches_path[index]
        category = sk_path.split(os.path.sep)[-2]

        pos_sample = sk_path.split('/')[-1].split('-')[:-1][0]
        pos_path = glob.glob(os.path.join(
            self.args.root, 'photo', category, pos_sample + '.*'))
        if len(pos_path) == 0:
            print(sk_path)
            return None

        pos_path = pos_path[0]
        photo_category = self.all_photos_path[category]
        photo_category = [p for p in photo_category if p != pos_path]

        neg_path = np.random.choice(photo_category)

        sk_data = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.args.max_size, self.args.max_size))
        img_data = ImageOps.pad(Image.open(pos_path).convert('RGB'), size=(self.args.max_size, self.args.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.args.max_size, self.args.max_size))

        sk_tensor = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)

        if self.mode == "train":
            pos_perm = generate_perm()
            neg_perm = generate_perm()
            
            img_jigsaw_tensor = permute_patch(img_tensor, pos_perm)
            sk_jigsaw_tensor = permute_patch(sk_tensor, pos_perm)
            neg_jigsaw_tensor = permute_patch(neg_tensor, neg_perm)
            
            sk_aug_tensor = self.aumentation(sk_data)
            img_aug_tensor = self.aumentation(img_data)
            return img_tensor, sk_tensor, img_aug_tensor, sk_aug_tensor, neg_tensor, self.all_categories.index(category), \
                img_jigsaw_tensor, sk_jigsaw_tensor, neg_jigsaw_tensor

        else:
            return sk_tensor, sk_path, img_tensor, pos_sample, self.all_categories.index(category)

if __name__ == "__main__":
    B = 4
    C = 3
    H = W = 224
    num_split = 4
    
    images = torch.randn(C, H, W)
    batch_perms = generate_perm(num_split=num_split)
    output = permute_patch(images, batch_perms, num_split=num_split)
    
    print(output.shape)