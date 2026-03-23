import os
import glob
import numpy as np
import torch
from src_fg.utils_fg import parse_sketchy_fg_photo, parse_sketchy_fg_sketch
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
        self.augmentation = augmented_transform()  # Strong augmentation for consistency
        self.return_orig = return_orig

        dataset_key = self.opts.dataset if hasattr(self.opts, 'dataset') else 'sketchy'
        unseen_classes = UNSEEN_CLASSES.get(dataset_key, UNSEEN_CLASSES['sketchy'])

        self.all_categories = sorted(os.listdir(os.path.join(self.opts.data_dir, 'sketch')))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')
            
        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories)*self.opts.data_split)]
            else:
                self.all_categories = sorted(set(self.all_categories) - set(used_cat))  # sorted!
        else:
            if mode == 'train':
                self.all_categories = sorted(set(self.all_categories) - set(unseen_classes))  # sorted!
            else:  # mode == 'val'
                self.all_categories = sorted(unseen_classes)  # sorted!

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
                self.all_sketches_path.extend(sorted(sketches))  # sorted!
                self.all_photos_path[category] = sorted(photos)  # sorted!
                valid_categories.append(category)
        
        # Update all_categories to only include valid ones (already sorted from above)
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
        
        # Apply strong augmentation for consistency regularization
        sk_aug_tensor = self.augmentation(sk_data)
        # Sketch-only augmentation ablation: keep photo branch un-augmented.
        img_aug_tensor = img_tensor
        
        if self.return_orig:
            return sk_tensor, img_tensor, neg_tensor, self.all_categories.index(category), filename, \
                sk_data, img_data, neg_data
        else:
            return sk_tensor, img_tensor, neg_tensor, sk_aug_tensor, img_aug_tensor, \
                   self.all_categories.index(category), filename

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms

def augmented_transform():
    """Strong augmentation for consistency regularization (CoPrompt-style)"""
    transform_list = [
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)

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
        self.all_categories = sorted(set(unseen_classes))

        self.paths = []
        for category in self.all_categories:
            if self.mode == "photo":
                self.paths.extend(sorted(glob.glob(os.path.join(self.args.data_dir, 'photo', category, '*'))))
            else:
                self.paths.extend(sorted(glob.glob(os.path.join(self.args.data_dir, 'sketch', category, '*'))))

    def __getitem__(self, index):
        filepath = self.paths[index]                
        category = filepath.split(os.path.sep)[-2]
        
        image = ImageOps.pad(Image.open(filepath).convert('RGB'),  size=(self.args.max_size, self.args.max_size))
        image_tensor = self.transform(image)
        
        return image_tensor, self.all_categories.index(category)
    
    def __len__(self):
        return len(self.paths)

class ValidDatasetFG(torch.utils.data.Dataset):
    def __init__(self, args, mode='photo'):
        super(ValidDatasetFG, self).__init__()
        self.args = args
        self.mode = mode
        self.transform = normal_transform()
        
        # Get unseen categories
        unseen_classes = UNSEEN_CLASSES.get('sketchy', UNSEEN_CLASSES['sketchy'])
        self.all_categories = sorted(set(unseen_classes))

        #Collect all file paths
        self.paths = []
        for category in self.all_categories:
            if self.mode == "photo":
                pattern = os.path.join(self.args.data_dir, 'photo', category, '*')
            else:
                pattern = os.path.join(self.args.data_dir, 'sketch', category, '*')

            category_files = sorted(glob.glob(pattern))
            self.paths.extend(category_files)


        self.labels = []
        self.filenames = []
        self.base_naems = []

        for path in self.paths:
            category = path.split(os.path.sep)[-2]
            cat_index = self.all_categories.index(category)
            self.labels.append(cat_index)

            filename = os.path.basename(path)
            self.filenames.append(filename)

            if self.mode == "photo":
                base_name = parse_sketchy_fg_photo(path)
            else:
                base_name = parse_sketchy_fg_sketch(path)
            self.base_naems.append(base_name)


    def __getitem__(self, index):
        filepath = self.paths[index]                
        category_index = self.labels[index]
        filename = self.filenames[index]
        base_name = self.base_naems[index]

        image = ImageOps.pad(
            Image.open(filepath).convert('RGB'),  
            size=(self.args.max_size, self.args.max_size)
        )

        image_tensor = self.transform(image)

        return image_tensor, category_index, filename, base_name
    
    def __len__(self):
        return len(self.paths)