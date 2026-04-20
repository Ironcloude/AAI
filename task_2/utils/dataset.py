"""Dataset class for the produce health/type classification task."""
# https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
import os
import random
from pathlib import Path
from typing import Callable
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torch.utils.data import  Dataset
from collections import Counter

class ProduceDataset(Dataset):
    """Aggregate produce dataset folder structure into a usable dataset.

    Supports binary classification of healthy vs rotten produce and
    produce-type labels for MTL objectives.
    """

    def __init__(self, dataset_root_dir: str | Path, 
                transform: Callable | None = None) -> None:
        """Initialise the dataset and assign labels from folder structure.

        Args:
            dataset_root_dir: Directory with all the images.
            transform: Optional transform to be applied on a sample.
        """
        self.dataset_root_dir = dataset_root_dir
        self.transform = transform
        self.image_paths = []
        self.health_lbls = []           # 0 = Healthy; 1 = Rotten
        self.produce_type_lbls = []     # int index per fruit type
        self.produce_type_names = []    # e.g. ["apple", "banana", ...]
        self.produce_type_to_index = {}
        self.assign_labels()

    def assign_labels(self) -> None:
        """Assign labels to images based on directory names.

        Health labels: 0 = Healthy, 1 = Rotten. 
        Produce type labels are assigned sequentially per produce type
        """
        # Iterate through each "<type>__<health>" directory.
        for folder_name in sorted(os.listdir(self.dataset_root_dir)):
            folder_path = os.path.join(self.dataset_root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Parse "<type>__<health>"  type="apple", health=0
            health_label = 0 if "healthy" in folder_name.lower() else 1
            fruit_type = folder_name.split(" __ ")[0].strip().lower()

            # Store produce type and corresponding index
            if fruit_type not in self.produce_type_to_index:
                self.produce_type_to_index[fruit_type] = len(self.produce_type_names)
                self.produce_type_names.append(fruit_type)

            type_label = self.produce_type_to_index[fruit_type]

            # Iterate over all images and assign produce type and health labels
            for image_name in os.listdir(folder_path):
                self.image_paths.append(os.path.join(folder_path, image_name))
                self.health_lbls.append(health_label)
                self.produce_type_lbls.append(type_label)
    
    def display_examples(self,num_samples: int = 5,
                         show_transformed: bool = False) -> None:
        """Display random samples from the dataset with their labels.

        Args:
            num_samples: Number of random samples to display.
            show_transformed: If True, apply the dataset's transform
                before display.
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
                print(f"Sample: {i} {'Rotten' if sample[1] == 1 else 'Healthy'}")
                plt.imshow(sample[0])
            if i == num_samples - 1:
                plt.show()
                break
        if not show_transformed:
            self.transform = original_transform

    def augment(self, sample):
        #TODO: Implement data augmentioned transformations (e.g. random horizontal flip, random rotation, etc.)
        # Part 4. https://medium.com/@ebimsv/mastering-cnns-in-pytorch-week-2-building-and-training-custom-and-pretrained-cnns-for-image-f040572c73c1
        pass

    def balance(self):
        #TODO: Implement data balancing - Weighted cross entropy loss or oversampling/undersampling techniques
        pass 

    def print_class_balance(self) -> None:
        """Print counts of health and type labels across the dataset."""
        print(f"\nLoaded {self.__len__()} images total\n")
        health_counts = Counter(self.health_lbls)
        type_counts = Counter(self.produce_type_lbls)
        print(f"Class Breakdown\n{'='*10}\n{'HEALTHY (0)':<20}{health_counts[0]}\n{'ROTTEN (1)':<20}{health_counts[1]}\n")
        print(f"\nType Breakdown\n{'='*10}")
        for type_idx, count in type_counts.items():
            print(f"{self.produce_type_names[type_idx].upper():<20}{count}")
        
    @property
    def num_produce_types(self) -> int:
        """Return the number of distinct produce types in the dataset."""
        return len(self.produce_type_names)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        """Return (sample, health_label, type_label) for the given index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_paths[idx]
        # Ensure image is RGB (some images have 4 channels)
        # TODO Ensure properly handling of different image types
        sample = Image.open(image_path).convert('RGBA').convert('RGB') 
        if self.transform:
            sample = self.transform(sample)

        return sample, self.health_lbls[idx], self.produce_type_lbls[idx]