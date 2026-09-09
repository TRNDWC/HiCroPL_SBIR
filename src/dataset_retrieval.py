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
        "tree", "wine_bottle", "deer", "chicken", "airplane", "wheelchair", 
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

# Extra SEEN classes mixed into GZS-SBIR (generalized ZS) evaluation, on top of
# UNSEEN_CLASSES. These classes were part of the training split, so their
# retrieval gallery/query images are ones the encoder already saw during
# training -- this tests robustness to seen-class distractors in the gallery,
# not held-out generalization on those specific images.
GENERALIZED_CLASSES = {
    "sketchy_ext": [
        "teapot",
        "harp",
        "piano",
        "trumpet",
        "saxophone",
        "hourglass",
        "mushroom",
        "pretzel",
        "bell",
    ],
    "sketchy_2": [
        "teapot",
        "harp",
        "piano",
        "trumpet",
        "saxophone",
        "hourglass",
        "mushroom",
        "pretzel",
        "bell",
    ],
    "tuberlin": [
        "blimp",
        "tablelamp",
        "telephone",
        "human-skeleton",
        "pickup truck",
    ],
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
        elif mode == 'train' and getattr(self.opts, 'cross_dataset_eval', False):
            # Cross-dataset ZS-SBIR: evaluation happens on a fully separate
            # dataset (see ValidDataset), so the within-dataset unseen split
            # doesn't need to be held out here -- train on every category.
            self.all_categories = sorted(self.all_categories)
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
    def __init__(self, args, mode='photo', base_data_dir=None):
        super(ValidDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.transform = normal_transform()

        # Across-dataset ZS-SBIR (--cross_dataset_eval): evaluate against a
        # dataset entirely different from the one trained on (e.g. train on
        # sketchy_ext, eval on tuberlin/quickdraw). base_data_dir (explicit
        # param, or args.eval_data_dir when the flag is set) overrides
        # args.data_dir, and dataset_key switches to args.eval_dataset so the
        # STANDARD unseen-test split of the target dataset is used (per the
        # paper protocol: "evaluate directly on the unseen test classes of
        # TU-Berlin-Ext and QuickDraw-Ext") -- not every category of that
        # dataset.
        cross_dataset_eval = getattr(self.args, 'cross_dataset_eval', False)
        if base_data_dir is None and cross_dataset_eval:
            base_data_dir = self.args.eval_data_dir
        self.data_dir = base_data_dir if base_data_dir is not None else self.args.data_dir

        if cross_dataset_eval:
            dataset_key = self.args.eval_dataset
        else:
            dataset_key = self.args.dataset if hasattr(self.args, 'dataset') else 'sketchy'
        unseen_classes = UNSEEN_CLASSES.get(dataset_key, UNSEEN_CLASSES['sketchy'])

        eval_mode_gzs = getattr(self.args, 'eval_mode_gzs', False)
        if eval_mode_gzs and cross_dataset_eval:
            raise ValueError("--eval_mode_gzs and --cross_dataset_eval are mutually exclusive.")
        if eval_mode_gzs and getattr(self.args, 'gzs_eval', False):
            raise ValueError("--eval_mode_gzs and --gzs_eval are mutually exclusive -- two different, "
                              "incompatible GZS mechanisms. --gzs_eval mixes a hand-picked SEEN-class "
                              "subset into both sketch and photo (non-standard). --eval_mode_gzs "
                              "implements the standard protocol, gallery = P^s (ALL train photos of "
                              "EVERY seen class) union P^u, query unchanged (S^u only).")

        if eval_mode_gzs:
            # Standard GZS-SBIR protocol: gallery = P^s union P^u, query = S^u
            # unchanged. P^s/P^u are read directly off disk (glob), independent
            # of the training DataLoader, per dataset_retrieval.py:127-153's
            # pattern for Sketchy.all_photos_path -- no subsampling.
            full_categories = sorted(os.listdir(os.path.join(self.data_dir, 'sketch')))
            if '.ipynb_checkpoints' in full_categories:
                full_categories.remove('.ipynb_checkpoints')
            seen_classes = sorted(set(full_categories) - set(unseen_classes))
            # Combined label vocabulary shared by BOTH the sketch (query) and
            # photo (gallery) ValidDataset instances, so a category's integer
            # label (self.all_categories.index(category) in __getitem__) is
            # IDENTICAL across both -- required for target = (photo_label ==
            # sketch_label) in model_hicropl.py to work once the gallery spans
            # two disjoint category sets: a seen-class gallery image can never
            # get the same label as any query (query labels only ever come
            # from unseen_classes), so it is correctly a distractor, never a
            # false positive.
            self.all_categories = sorted(set(unseen_classes) | set(seen_classes))

            self.paths = []
            if self.mode == 'photo':
                paths_unseen = []
                for category in sorted(unseen_classes):
                    paths_unseen.extend(sorted(glob.glob(os.path.join(self.data_dir, 'photo', category, '*'))))
                paths_seen = []
                for category in seen_classes:
                    paths_seen.extend(sorted(glob.glob(os.path.join(self.data_dir, 'photo', category, '*'))))
                self.paths = paths_seen + paths_unseen
                self.n_gallery_seen = len(paths_seen)
                self.n_gallery_unseen = len(paths_unseen)
                print(f"GZS_FP | on=1 | n_gallery_seen={self.n_gallery_seen} | "
                      f"n_gallery_unseen={self.n_gallery_unseen} | "
                      f"n_gallery_total={self.n_gallery_seen + self.n_gallery_unseen}")
            else:
                for category in sorted(unseen_classes):
                    self.paths.extend(sorted(glob.glob(os.path.join(self.data_dir, 'sketch', category, '*'))))
                self.n_query = len(self.paths)
                print(f"GZS_FP | on=1 | n_query={self.n_query}")
            return

        # GZS-SBIR (non-standard, existing flag): mix a fixed set of SEEN
        # classes into the eval gallery/query on top of the unseen ones. See
        # GENERALIZED_CLASSES docstring for the train/test image-leakage
        # caveat this implies for the seen classes. Mutually exclusive with
        # --cross_dataset_eval (enforced in the training script), so
        # dataset_key here always refers to args.dataset.
        if getattr(self.args, 'gzs_eval', False):
            seen_classes = GENERALIZED_CLASSES.get(dataset_key, [])
            if len(seen_classes) == 0:
                print(f"[WARN] --gzs_eval set but no GENERALIZED_CLASSES entry for dataset '{dataset_key}'; "
                      f"falling back to unseen-only evaluation.")
            self.all_categories = sorted(set(unseen_classes) | set(seen_classes))
        else:
            self.all_categories = sorted(set(unseen_classes))

        self.paths = []
        for category in self.all_categories:
            if self.mode == "photo":
                self.paths.extend(sorted(glob.glob(os.path.join(self.data_dir, 'photo', category, '*'))))
            else:
                self.paths.extend(sorted(glob.glob(os.path.join(self.data_dir, 'sketch', category, '*'))))

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