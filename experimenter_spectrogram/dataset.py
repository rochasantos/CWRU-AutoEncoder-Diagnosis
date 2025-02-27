import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset para espectrogramas.

        Args:
            root_dir (str): Diretório com subpastas de classes contendo imagens.
            transform: Transformações aplicadas às imagens.
        """
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),  # Ajusta para entrada de redes pré-treinadas
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalização
        ])
        self.samples = self._load_file_paths()

    def _load_file_paths(self):
        """
        Carrega os caminhos das imagens e as classes.
        """
        samples = []
        class_names = sorted(os.listdir(self.root_dir))  # Assume que cada pasta é uma classe
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith(".png") or filename.endswith(".jpg"):
                        samples.append((os.path.join(class_dir, filename), class_to_idx[class_name]))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
