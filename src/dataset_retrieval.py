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

    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False,
                 transform_aug_photo=None, transform_aug_sketch=None):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig
        # Augmentation branch (on unless --disable_aug_branch). Both must be
        # supplied together; when absent __getitem__ keeps its original arity so
        # every existing unpack site is unaffected.
        self.transform_aug_photo = transform_aug_photo
        self.transform_aug_sketch = transform_aug_sketch

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

        sk_data  = Image.open(sk_path).convert('RGB')
        img_data = Image.open(img_path).convert('RGB')
        neg_data = Image.open(neg_path).convert('RGB')

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        if self.return_orig:
            return sk_tensor, img_tensor, neg_tensor, self.all_categories.index(category), filename, \
                sk_data, img_data, neg_data

        if self.transform_aug_photo is None:
            return sk_tensor, img_tensor, neg_tensor, self.all_categories.index(category), filename

        # Augmented views go LAST so the leading 5 entries keep their meaning.
        # They are also drawn AFTER every deterministic transform above, so the
        # extra RNG they consume cannot shift anything that came before -- the
        # non-augmented part of the sample is bit-identical to a run with
        # --disable_aug_branch under the same seed.
        sk_aug_tensor = self.transform_aug_sketch(sk_data)
        img_aug_tensor = self.transform_aug_photo(img_data)
        return sk_tensor, img_tensor, neg_tensor, self.all_categories.index(category), filename, \
            sk_aug_tensor, img_aug_tensor

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms

    @staticmethod
    def data_transform_aug_photo(opts):
        """Augmented view for the photo branch -- close to the MoCo v2 / DINO recipe.

        Crop scale is 0.4 (not MoCo's 0.2) on purpose: the target of the InfoNCE
        term is a FROZEN CLIP, not a co-adapting encoder. A crop so aggressive
        that the subject is gone still produces a confident frozen feature, and
        the loss would drag the trainable branch toward that noise. In SSL both
        views adapt together, so the failure mode does not arise there.

        GaussianBlur/Solarization are dropped: the backbone is ViT-B/32, whose
        32x32 patches barely register mild blur.

        --aug_identity_transform (Run B) short-circuits all of that and returns
        data_transform itself -- reused, not re-declared, so photo_aug comes out
        bit-wise equal to photo. That leaves clip_aug in place while removing the
        perturbation, isolating the second encoder as the only variable.
        """
        if getattr(opts, 'aug_identity_transform', False):
            return Sketchy.data_transform(opts)
        return transforms.Compose([
            transforms.RandomResizedCrop(opts.max_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def data_transform_aug_sketch(opts):
        """Augmented view for the sketch branch -- deliberately NOT the photo recipe.

        Sketches are black strokes on white with R==G==B, which makes three of
        the five standard colour ops mathematically no-ops (measured on a
        synthetic sketch: saturation 0.000%, hue 0.000%, RandomGrayscale 0.000%
        pixel change). Only brightness/contrast do anything, so the rest is
        replaced by geometric jitter -- the direction the sketch literature
        takes instead of colour.

        GaussianBlur is excluded outright: at sigma=2.0 (the top of MoCo's
        [0.1, 2.0] range) 99% of stroke pixels fall below the ink threshold.
        Crop scale is milder than photo's since strokes are sparse and an
        aggressive crop easily lands on blank canvas. fill=255 keeps the
        canvas white where RandomAffine exposes new area.

        --aug_identity_transform (Run B): same short-circuit as the photo side,
        sketch_aug comes out bit-wise equal to sketch.
        """
        if getattr(opts, 'aug_identity_transform', False):
            return Sketchy.data_transform(opts)
        return transforms.Compose([
            transforms.RandomResizedCrop(opts.max_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08),
                                    scale=(0.9, 1.1), fill=255),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
        
        image = Image.open(filepath).convert('RGB')
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

        image = Image.open(filepath).convert('RGB')

        image_tensor = self.transform(image)

        return image_tensor, category_index, filename, base_name
    
    def __len__(self):
        return len(self.paths)