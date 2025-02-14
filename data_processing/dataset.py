import os
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image

class SpectrogramPairDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        """
        Dataset that loads pairs of spectrograms from two distinct directories.
        Each input image has a corresponding image in the target.
        
        Parameters:
        - input_dir: Directory containing the input spectrograms.
        - target_dir: Directory contining the output spectrograms.
        - transform: Transformations applied to images.
        """
        self.input_dataset = ImageFolder(root=input_dir, transform=transform)
        self.target_dataset = ImageFolder(root=target_dir, transform=transform)

        self.input_classes = self.input_dataset.classes  
        self.target_classes = self.target_dataset.classes  

        assert self.input_classes == self.target_classes # checks if the classes between input and output match 

        # Create index mapping by class
        self.input_class_to_indices = self._get_class_indices(self.input_dataset)
        self.target_class_to_indices = self._get_class_indices(self.target_dataset)

        self.transform = transform

    def _get_class_indices(self, dataset):
        """
        Returns a dictionary that associates each class with a list of indices of the images belonging to it.
        """
        class_to_indices = {}
        for idx, (_, class_id) in enumerate(dataset.samples):
            if class_id not in class_to_indices:
                class_to_indices[class_id] = []
            class_to_indices[class_id].append(idx)
        return class_to_indices

    def __len__(self):
        return len(self.input_dataset)

    def __getitem__(self, index):
        """
        Returns a pair (input_img, target_img) ensuring they belong to the same class.
        """
        input_img, class_id = self.input_dataset[index]  # Obtém a imagem de entrada e sua classe

        # Gets a matching index in target_dir for the same class
        target_index = self.target_class_to_indices[class_id][index % len(self.target_class_to_indices[class_id])]
        target_img, _ = self.target_dataset[target_index]

        return input_img, target_img  # Returns the matching pair


class PairedImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        """
        Custom PyTorch dataset that loads paired images from two different datasets.

        Args:
        - input_dir (str): Path to the directory containing input images.
        - target_dir (str): Path to the directory containing target images.
        - transform (callable, optional): Transformations to apply to both images.
        """
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

        # Get sorted list of class names (folders)
        self.input_classes = sorted(os.listdir(input_dir))
        self.target_classes = sorted(os.listdir(target_dir))

        assert self.input_classes == self.target_classes, "Class directories in input and target must match!"

        # Create dictionary mapping each class to its image paths
        self.input_images = self._load_images(input_dir)
        self.target_images = self._load_images(target_dir)

        # Ensure both datasets have the same number of images per class
        self.image_pairs = self._pair_images()

    def _load_images(self, root_dir):
        """Loads images from a directory and maps them to class labels."""
        image_dict = {}
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                image_dict[class_name] = sorted(
                    [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
                )
        return image_dict

    def _pair_images(self):
        """Pairs images from input and target directories based on class labels."""
        pairs = []
        for class_name in self.input_classes:
            input_list = self.input_images[class_name]
            target_list = self.target_images[class_name]

            # Ensure lists are the same size (truncate if necessary)
            min_len = min(len(input_list), len(target_list))
            for i in range(min_len):
                pairs.append((input_list[i], target_list[i]))

        return pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        """Returns a pair of images (input, target)."""
        input_path, target_path = self.image_pairs[idx]

        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img