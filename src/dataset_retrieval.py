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
    # 30 lớp test của TUBerlin Ext (ZSE-SBIR split, dùng cho within-dataset ZS-SBIR).
    # Khi cross_dataset_eval (S→T), ValidDataset sẽ lọc thêm các lớp trùng với
    # source train (Sketchy 100) → còn lại 21 lớp unseen đúng theo bài báo.
    "tuberlin": [
        "ant", "banana", "bottle opener", "brain", "bread", "bridge",
        "bus", "canoe", "fan", "frying-pan", "horse", "hot air balloon",
        "laptop", "lighter", "parachute", "penguin", "pizza", "rollerblades",
        "shoe", "snowboard", "space shuttle", "streetlight", "suitcase",
        "t-shirt", "table", "teacup", "telephone", "tractor", "trombone",
        "windmill"
    ],
    # 30 lớp test của QuickDraw Ext (ZSE-SBIR split, dùng cho within-dataset ZS-SBIR).
    # Khi cross_dataset_eval (S→Q), ValidDataset sẽ lọc thêm các lớp trùng với
    # source train (Sketchy 100) → còn lại 11 lớp unseen đúng theo bài báo.
    "quickdraw": [
        "banana", "bat", "beach", "bread", "cactus", "cake", "campfire",
        "cow", "dolphin", "door", "fan", "feather", "fire_hydrant", "frog",
        "giraffe", "hamburger", "helicopter", "megaphone", "mouse",
        "palm tree", "raccoon", "rhinoceros", "saw", "scissors", "shark",
        "skyscraper", "tiger", "tree", "windmill", "zebra"
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

        if cross_dataset_eval:
            # Zero-shot guarantee: loại bỏ các lớp trong tập test của dataset đích
            # mà đã xuất hiện trong tập train của dataset nguồn (args.data_dir).
            # Normalize tên để xử lý bất đồng định dạng, ví dụ:
            #   Sketchy "hot-air_balloon" == TUBerlin "hot air balloon"
            def _norm_cls(s):
                return s.lower().replace('-', ' ').replace('_', ' ').strip()

            src_sketch_dir = os.path.join(self.args.data_dir, 'sketch')
            src_train_norm = set()
            if os.path.isdir(src_sketch_dir):
                src_cats = os.listdir(src_sketch_dir)
                if '.ipynb_checkpoints' in src_cats:
                    src_cats.remove('.ipynb_checkpoints')
                src_train_norm = {_norm_cls(c) for c in src_cats}

            before_filter = list(unseen_classes)
            unseen_classes = [c for c in unseen_classes if _norm_cls(c) not in src_train_norm]
            removed = sorted(set(before_filter) - set(unseen_classes))
            print(f"[cross_dataset_eval] Lọc lớp trùng với source train: "
                  f"{len(before_filter)} → {len(unseen_classes)} lớp "
                  f"(loại {len(removed)}: {removed})")

        eval_mode_gzs = getattr(self.args, 'eval_mode_gzs', False)
        if eval_mode_gzs and cross_dataset_eval:
            raise ValueError("--eval_mode_gzs and --cross_dataset_eval are mutually exclusive.")
        if eval_mode_gzs and getattr(self.args, 'gzs_eval', False):
            raise ValueError("--eval_mode_gzs and --gzs_eval are mutually exclusive -- two different, "
                              "incompatible GZS mechanisms. --gzs_eval mixes a hand-picked SEEN-class "
                              "subset into both sketch and photo (non-standard). --eval_mode_gzs "
                              "implements the SEM-PCYC protocol: gallery = unseen photos + random "
                              "20% seen photos; query = unseen sketches + random 20% seen sketches.")

        if eval_mode_gzs:
            # GZS-SBIR protocol following SEM-PCYC (Dutta & Akata, CVPR 2019):
            # Both gallery and query are augmented with a random subset of seen
            # data.  The number of seen samples added = perc × number of unseen
            # samples in that modality (perc = 0.2, i.e. 20%).
            #
            # Gallery = unseen photos  +  random_sample(seen photos, 0.2 × |unseen photos|)
            # Query   = unseen sketches + random_sample(seen sketches, 0.2 × |unseen sketches|)
            perc = 0.2
            full_categories = sorted(os.listdir(os.path.join(self.data_dir, 'sketch')))
            if '.ipynb_checkpoints' in full_categories:
                full_categories.remove('.ipynb_checkpoints')
            seen_classes = sorted(set(full_categories) - set(unseen_classes))
            # Combined label vocabulary shared by BOTH the sketch (query) and
            # photo (gallery) ValidDataset instances, so a category's integer
            # label (self.all_categories.index(category) in __getitem__) is
            # IDENTICAL across both -- required for target = (photo_label ==
            # sketch_label) in model_hicropl.py to work correctly.
            self.all_categories = sorted(set(unseen_classes) | set(seen_classes))

            self.paths = []
            if self.mode == 'photo':
                paths_unseen = []
                for category in sorted(unseen_classes):
                    paths_unseen.extend(sorted(glob.glob(os.path.join(self.data_dir, 'photo', category, '*'))))
                paths_seen_all = []
                for category in seen_classes:
                    paths_seen_all.extend(sorted(glob.glob(os.path.join(self.data_dir, 'photo', category, '*'))))
                # Sample 20% of the unseen count from seen photos
                n_sample = int(perc * len(paths_unseen))
                n_sample = min(n_sample, len(paths_seen_all))
                rng = np.random.RandomState(42)  # fixed seed for reproducibility
                idx = rng.choice(len(paths_seen_all), n_sample, replace=False)
                idx.sort()
                paths_seen_sampled = [paths_seen_all[i] for i in idx]
                self.paths = paths_seen_sampled + paths_unseen
                self.n_gallery_seen = len(paths_seen_sampled)
                self.n_gallery_unseen = len(paths_unseen)
                print(f"GZS_FP | on=1 | protocol=SEM-PCYC | perc={perc} | "
                      f"n_seen_pool={len(paths_seen_all)} | "
                      f"n_gallery_seen={self.n_gallery_seen} | "
                      f"n_gallery_unseen={self.n_gallery_unseen} | "
                      f"n_gallery_total={self.n_gallery_seen + self.n_gallery_unseen}")
            else:
                paths_unseen = []
                for category in sorted(unseen_classes):
                    paths_unseen.extend(sorted(glob.glob(os.path.join(self.data_dir, 'sketch', category, '*'))))
                paths_seen_all = []
                for category in seen_classes:
                    paths_seen_all.extend(sorted(glob.glob(os.path.join(self.data_dir, 'sketch', category, '*'))))
                # Sample 20% of the unseen count from seen sketches
                n_sample = int(perc * len(paths_unseen))
                n_sample = min(n_sample, len(paths_seen_all))
                rng = np.random.RandomState(42)  # same seed for reproducibility
                idx = rng.choice(len(paths_seen_all), n_sample, replace=False)
                idx.sort()
                paths_seen_sampled = [paths_seen_all[i] for i in idx]
                self.paths = paths_seen_sampled + paths_unseen
                self.n_query_seen = len(paths_seen_sampled)
                self.n_query_unseen = len(paths_unseen)
                self.n_query = len(self.paths)
                print(f"GZS_FP | on=1 | protocol=SEM-PCYC | perc={perc} | "
                      f"n_seen_pool={len(paths_seen_all)} | "
                      f"n_query_seen={self.n_query_seen} | "
                      f"n_query_unseen={self.n_query_unseen} | "
                      f"n_query_total={self.n_query}")
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