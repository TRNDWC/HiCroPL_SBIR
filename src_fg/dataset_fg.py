import os
import glob
import numpy as np
import torch


from torchvision import transforms
from PIL import Image, ImageOps

UNSEEN_CLASSES = {
    "sketchy_2": [
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
        "window",
    ],
}


        
def normal_transform():
    dataset_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return dataset_transforms


class SketchyDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        self.root = getattr(self.args, "root", getattr(self.args, "data_dir", None))
        if self.root is None:
            raise ValueError("SketchyDataset requires args.root or args.data_dir")
        unseen_classes = UNSEEN_CLASSES["sketchy_2"]

        self.all_categories = os.listdir(
            os.path.join(self.root, 'sketch'))
        self.transform = normal_transform()

        if self.mode == "train":
            self.all_categories = list(
                set(self.all_categories) - set(unseen_classes))
        else:
            self.all_categories = list(set(unseen_classes))

        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            self.all_sketches_path.extend(
                glob.glob(os.path.join(self.root, 'sketch', category, '*')))
            self.all_photos_path[category] = glob.glob(
                os.path.join(self.root, 'photo', category, 'n*'))

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, index):
        sk_path = self.all_sketches_path[index]
        category = sk_path.split(os.path.sep)[-2]

        pos_sample = sk_path.split('/')[-1].split('-')[:-1][0]
        pos_path = glob.glob(os.path.join(
            self.root, 'photo', category, pos_sample + '.*'))
        if len(pos_path) == 0:
            print(sk_path)
            return None

        pos_path = pos_path[0]
        photo_category = self.all_photos_path[category]
        photo_category = [p for p in photo_category if p != pos_path]

        neg_path = np.random.choice(photo_category)

        sk_data = Image.open(sk_path).convert('RGB')
        img_data = Image.open(pos_path).convert('RGB')
        neg_data = Image.open(neg_path).convert('RGB')

        sk_tensor = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)

        if self.mode == "train":
            return img_tensor, sk_tensor, neg_tensor, self.all_categories.index(category)

        else:
            return sk_tensor, sk_path, img_tensor, pos_sample, self.all_categories.index(category)

if __name__ == "__main__":
    print("dataset_fg module loaded")
