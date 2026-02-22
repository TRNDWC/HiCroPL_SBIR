import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps

# Unseen classes for different datasets (ZS-SBIR evaluation)
UNSEEN_CLASSES = {
    "sketchy": [
        "bat", "cabin", "cow", "dolphin", "door", "giraffe", "helicopter",
        "mouse", "pear", "raccoon", "rhinoceros", "saw", "scissors",
        "seagull", "skyscraper", "songbird", "sword", "tree", "wheelchair",
        "windmill", "window"
    ],
    "sketchy_ext": [
        "bat", "cabin", "cow", "dolphin", "door", "giraffe", "helicopter",
        "mouse", "pear", "raccoon", "rhinoceros", "saw", "scissors",
        "seagull", "skyscraper", "songbird", "sword", "tree", "wheelchair",
        "windmill", "window"
    ],
    "sketchy_1": [
        "cup", "swan", "harp", "squirrel", "snail", "ray", "pineapple",
        "volcano", "rifle", "scissors", "parrot", "windmill", "teddy_bear",
        "tree", "wine_bottle", "deer", "chicken", "hotdog", "wheelchair",
        "tank", "umbrella", "butterfly", "camel", "horse", "bell"
    ],
    "sketchy_2": [
        "bat", "cabin", "cow", "dolphin", "door", "giraffe", "helicopter",
        "mouse", "pear", "raccoon", "rhinoceros", "saw", "scissors",
        "seagull", "skyscraper", "songbird", "sword", "tree", "wheelchair",
        "windmill", "window"
    ],
    "tuberlin": [
        "helicopter", "wrist-watch", "mermaid", "mosquito", "pear", "couch",
        "hammer", "purse", "house", "tennis-racket", "toilet", "panda",
        "butterfly", "mug", "wineglass", "motorbike", "eyeglasses",
        "hot air balloon", "screwdriver", "skull", "truck", "palm tree",
        "cell phone", "horse", "sailboat", "suv", "church", "floor lamp",
        "pipe (for smoking)", "tv"
    ],
    "quickdraw": [
        "airplane", "alarm_clock", "ant", "apple", "axe", "banana", "bat",
        "bear", "bee", "bench", "bicycle", "bread", "bus", "butterfly",
        "cactus", "cake", "camel", "candle", "car", "castle", "cat", "chair",
        "church", "couch", "cow", "crab", "crocodilian", "dolphin",
        "eyeglasses", "guitar"
    ]
}

class Sketchy(torch.utils.data.Dataset):

    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig

        dataset_key = self.opts.dataset if hasattr(self.opts, 'dataset') else 'sketchy'
        unseen_classes = UNSEEN_CLASSES.get(dataset_key, UNSEEN_CLASSES['sketchy'])

        self.all_categories = os.listdir(os.path.join(self.opts.data_dir, 'sketch'))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')
            
        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories)*self.opts.data_split)]
            else:
                self.all_categories = list(set(self.all_categories) - set(used_cat))
        else:
            if mode == 'train':
                self.all_categories = list(set(self.all_categories) - set(unseen_classes))
            else:  # mode == 'val'
                self.all_categories = unseen_classes

        self.all_sketches_path = []
        self.all_photos_path = {}
        valid_categories = []

        for category in self.all_categories:
            # Try multiple extensions for sketches
            sketches = glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.png'))
            if len(sketches) == 0:
                sketches = glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*'))
            
            # Try multiple extensions for photos
            photos = glob.glob(os.path.join(self.opts.data_dir, 'photo', category, '*.jpg'))
            if len(photos) == 0:
                photos = glob.glob(os.path.join(self.opts.data_dir, 'photo', category, '*.png'))
            if len(photos) == 0:
                photos = glob.glob(os.path.join(self.opts.data_dir, 'photo', category, '*.jpeg'))
            if len(photos) == 0:
                photos = glob.glob(os.path.join(self.opts.data_dir, 'photo', category, '*'))
            
            # Debug: print categories that don't have data
            if len(sketches) == 0 or len(photos) == 0:
                if mode == 'val':  # Only print for validation to debug
                    print(f"Skipping category '{category}': {len(sketches)} sketches, {len(photos)} photos")
            
            # Only add category if both sketches and photos exist
            if len(sketches) > 0 and len(photos) > 0:
                self.all_sketches_path.extend(sketches)
                self.all_photos_path[category] = photos
                valid_categories.append(category)
        
        # Update all_categories to only include valid ones
        self.all_categories = valid_categories

    def __len__(self):
        return len(self.all_sketches_path)
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]
        filename = os.path.basename(filepath)
        
        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)

        sk_path  = filepath
        img_path = np.random.choice(self.all_photos_path[category])
        neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        if self.return_orig:
            return sk_tensor, img_tensor, neg_tensor, self.all_categories.index(category), filename, \
                sk_data, img_data, neg_data
        else:
            return sk_tensor, img_tensor, neg_tensor, self.all_categories.index(category), filename

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms

def normal_transform():
    dataset_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return dataset_transforms

class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='photo'):
        super(ValidDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.transform = normal_transform()
        
        dataset_key = self.args.dataset if hasattr(self.args, 'dataset') else 'sketchy'
        unseen_classes = UNSEEN_CLASSES.get(dataset_key, UNSEEN_CLASSES['sketchy'])
        self.all_categories = list(set(unseen_classes))

        self.paths = []
        for category in self.all_categories:
            if self.mode == "photo":
                self.paths.extend(glob.glob(os.path.join(self.args.data_dir, 'photo', category, '*')))
            else:
                self.paths.extend(glob.glob(os.path.join(self.args.data_dir, 'sketch', category, '*')))

    def __getitem__(self, index):
        filepath = self.paths[index]                
        category = filepath.split(os.path.sep)[-2]
        
        image = ImageOps.pad(Image.open(filepath).convert('RGB'),  size=(self.args.max_size, self.args.max_size))
        image_tensor = self.transform(image)
        
        return image_tensor, self.all_categories.index(category)
    
    def __len__(self):
        return len(self.paths)

