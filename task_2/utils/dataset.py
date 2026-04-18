

#  Class for building Dataset object
# https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
import os
import random
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'}


class ProduceDataset(Dataset):
    "Produce dataset for binary classification of healthy vs rotten produce"

    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            augment_train (bool): If True, apply data augmentation before the base transform.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.assign_labels()

    @staticmethod
    def _build_augment_transform():
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomGrayscale(p=0.15),  # 10% chance — forces texture learning over colour
        ])

    def assign_labels(self):
        """
        Assigns labels to images based on directory names (Healthy vs Rotten)
        0 = Healthy; 1 = Rotten
        Uses sorted() for deterministic ordering across runs.
        """
        for folder_name in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                label = 0 if "Healthy" in folder_name else 1
                for img_name in sorted(os.listdir(folder_path)):
                    ext = os.path.splitext(img_name)[1]
                    if ext in VALID_EXTENSIONS:
                        self.image_paths.append(os.path.join(folder_path, img_name))
                        self.labels.append(label)
    
    def display_examples(self, num_samples=5, show_transformed=False):
        """
        Displays random n samples from the dataset with their corresponding labels.

        Autotransforms nromalise
        """
        if not show_transformed:
            # Temporarily disable transform for display
            original_transform = self.transform
            self.transform = None

        random_indices = random.sample(range(len(self)), num_samples)
        random_samples = [self[id] for id in random_indices]
        for i, sample in enumerate(random_samples):
            ax = plt.subplot(1, num_samples, i + 1)
            plt.tight_layout()
            ax.set_title('Sample {}'.format(i))
            ax.axis('off')
            if show_transformed:
                print("Sample: ", i, sample[0].shape, sample[1])
                # Pytorch transforms images to C, H, W. Matplot expects H, W, C => Permute for display
                # i.e [3, 384, 384] -> [384, 384, 3]
                plt.imshow(sample[0].permute(1, 2, 0))
            else:
                print("Sample: ", i, sample[1])
                plt.imshow(sample[0])
            if i == num_samples - 1:
                plt.show()
                break
        if not show_transformed:
            self.transform = original_transform

    def augment(self, sample):
        """Apply augmentation transforms to a PIL image. Only active when augment_train=True."""
        if self._augment_transform is not None:
            return self._augment_transform(sample)
        return sample
    
    def _refresh(self):
        """Clears and rebuilds image_paths and labels from current state of disk."""
        self.image_paths = []
        self.labels = []
        self.assign_labels()
        print(f"Dataset refreshed: {len(self.image_paths)} images remaining.")

    def remove_duplicates(self, dry_run=True):
        """
        Detects exact/near-duplicate images using perceptual hashing and removes
        extras from disk. Keeps one image per duplicate group.

        Args:
            dry_run (bool): If True, prints what would be deleted without removing anything.
        """
        import imagehash
        from collections import defaultdict
        from tqdm import tqdm

        print("Hashing images for duplicate detection...")
        hash_map = defaultdict(list)
        for path in tqdm(self.image_paths):
            try:
                h = str(imagehash.phash(Image.open(path).convert('RGB')))
                hash_map[h].append(path)
            except Exception as e:
                print(f"  Could not open {path}: {e}")

        to_delete = []
        for paths in hash_map.values():
            if len(paths) > 1:
                to_delete.extend(paths[1:])  # keep paths[0], delete the rest

        print(f"Duplicate groups: {len([v for v in hash_map.values() if len(v) > 1])}")
        print(f"Files to delete:  {len(to_delete)}")

        if dry_run:
            print("\nDry run — no files deleted. Sample:")
            for p in to_delete[:10]:
                print(f"  {p}")
            return to_delete

        for path in to_delete:
            try:
                os.remove(path)
            except Exception as e:
                print(f"  Could not delete {path}: {e}")

        print(f"Deleted {len(to_delete)} duplicate files.")
        self._refresh()
        return to_delete

    def remove_augmented(self, dry_run=True):
        """
        Detects pre-augmented copies (flipped/rotated variants of existing images)
        within each folder and removes them from disk.

        Args:
            dry_run (bool): If True, prints what would be deleted without removing anything.
        """
        import imagehash
        from collections import defaultdict
        from tqdm import tqdm

        TRANSFORMS = {
            'flip_h':     lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
            'flip_v':     lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
            'rotate_90':  lambda img: img.rotate(90),
            'rotate_180': lambda img: img.rotate(180),
        }

        by_folder = defaultdict(list)
        for path in self.image_paths:
            by_folder[os.path.dirname(path)].append(path)

        augmented_copies = set()

        for folder, paths in tqdm(by_folder.items(), desc="Scanning folders"):
            folder_hashes = {}
            for path in paths:
                try:
                    h = str(imagehash.phash(Image.open(path).convert('RGB')))
                    folder_hashes[h] = path
                except:
                    pass

            for path in paths:
                if path in augmented_copies:
                    continue
                try:
                    img = Image.open(path).convert('RGB')
                except:
                    continue
                for transform_fn in TRANSFORMS.values():
                    try:
                        h = str(imagehash.phash(transform_fn(img)))
                        if h in folder_hashes and folder_hashes[h] != path:
                            augmented_copies.add(folder_hashes[h])
                    except:
                        pass

        print(f"\nPre-augmented copies found: {len(augmented_copies)}")

        if dry_run:
            print("Dry run — no files deleted. Sample:")
            for p in list(augmented_copies)[:10]:
                print(f"  {p}")
            return list(augmented_copies)

        for path in augmented_copies:
            try:
                os.remove(path)
            except Exception as e:
                print(f"  Could not delete {path}: {e}")

        print(f"Deleted {len(augmented_copies)} pre-augmented files.")
        self._refresh()
        return list(augmented_copies)


    def print_class_balance(self):
        label_counts = Counter(self.labels)
        print(f"Class Breakdown\n{'='*10}\nHEALTHY (0)\t{label_counts[0]}\nROTTEN (1)\t{label_counts[1]}\n")
        

    def __len__(self):
        """Allows len(dataset) to return the size of the dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Allow support for indexing, e.g. dataset[0] to return the first sample and corresponding label."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_paths[idx]
        sample = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label
