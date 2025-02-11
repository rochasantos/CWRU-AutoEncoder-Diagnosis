from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

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
