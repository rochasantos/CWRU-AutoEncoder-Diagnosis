import os
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio

class UOREDSignalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Lista arquivos .mat
        self.files = [f for f in os.listdir(root_dir) if f.endswith(".npy")]
        self.files.sort()

        # Extrai classes
        self.classes = sorted(set(f.split("_")[0] for f in self.files))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Labels como inteiros para cada amostra
        self.targets = [self.class_to_idx[f.split("_")[0]] for f in self.files]

    @property
    def num_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        path = os.path.join(self.root_dir, file_name)

        # Carrega .mat
        signal = np.load(path).squeeze()
        
        if signal is None:
            raise KeyError(f"Nenhum ndarray válido em {path}")

        # Aplica transform opcional
        if self.transform:
            signal = self.transform(signal)

        # Converte para tensor
        signal = torch.tensor(signal, dtype=torch.float32)

        # Ensure torch tensor and channel-first shape
        if not isinstance(signal, torch.Tensor):
            signal = torch.from_numpy(np.asarray(signal))
        signal = signal.float()
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)

        # Label como tensor long
        label = torch.tensor(self.targets[idx]).long()

        return signal, label
