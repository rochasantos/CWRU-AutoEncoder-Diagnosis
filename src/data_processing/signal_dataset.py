import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SignalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Caminho para o diretório que contém as pastas das classes (N, I, O, B).
            transform (callable, optional): Transformação a ser aplicada nos dados.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []

        # Mapeia os nomes dos diretórios (classes) para índices numéricos
        self.class_to_index = {'I': 0, 'O': 1, 'B': 2}

        # Percorre os diretórios e coleta os caminhos dos arquivos e suas labels
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.npy'):
                        file_path = os.path.join(class_dir, file_name)
                        self.file_paths.append(file_path)
                        # Usa o nome do diretório como label
                        self.labels.append(self.class_to_index[class_name])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Carrega o arquivo .npy
        signal = np.load(self.file_paths[idx])
        label = self.labels[idx]

        # Converte para tensor
        signal = torch.from_numpy(signal).float()
        label = torch.tensor(label, dtype=torch.long)

        # Aplica transformações, se houver
        if self.transform:
            signal = self.transform(signal)

        return signal, label