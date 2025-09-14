# src/data_processing/vibration_dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class VibrationDataset(Dataset):
    def __init__(self, root_dir, classes=("N", "I", "O", "B"), transform=None):
        self.root_dir = root_dir
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform  # store user transform

        self.samples = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.endswith(".npy"):
                    self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No .npy files found under {self.root_dir} with classes {self.classes}")

        # Expose labels for external samplers/splitters
        self.labels = [label for _, label in self.samples]
        self.targets = self.labels  # common alias in PyTorch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path)  # expects 1D array [L]

        # Apply optional transform (accepts np.ndarray or torch.Tensor)
        if self.transform is not None:
            x = self.transform(x)

        # Ensure torch tensor and channel-first shape
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.asarray(x))
        x = x.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)  # -> [1, L]

        return x, torch.tensor(label).long()
