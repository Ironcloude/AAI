

#  Class for building Dataset object
# https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
import os
import random
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torch.utils.data import  Dataset
from collections import Counter

class ProduceDataset(Dataset):
    "Produce dataset for binary classification of healthy vs rotten produce"

    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.assign_labels()

    def assign_labels(self):
        """
        Assigns labels to images based on directory names (Healthy vs Rotten)
        0 = Healthy; 1 = Rotten        
        """
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                label = 0 if "Healthy" in folder_name else 1
                
                for img_name in os.listdir(folder_path):
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
        #TODO: Implement data augmentioned transformations (e.g. random horizontal flip, random rotation, etc.)
        # Part 4. https://medium.com/@ebimsv/mastering-cnns-in-pytorch-week-2-building-and-training-custom-and-pretrained-cnns-for-image-f040572c73c1
        pass

    def balance(self):
        #TODO: Implement data balancing - Weighted cross entropy loss or oversampling/undersampling techniques
        pass 

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
        # Ensure image is RGB (some images have 4 channels)
        sample = Image.open(image_path).convert('RGBA').convert('RGB') 
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label