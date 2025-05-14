import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit


class VibrationDataset(Dataset):
    def __init__(self, data_dir, transform=None, class_map=None):
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npy")
        ])
        self.transform = transform

        # Map string labels to integers
        self.class_map = class_map or self._build_class_map()

        self.labels = []
        for f in self.files:
            sample = np.load(f, allow_pickle=True).item()
            label = sample["label"]
            label = self.class_map[label] if isinstance(label, str) else label
            self.labels.append(label)

    def _build_class_map(self):
        """Automatically builds class_map from available labels"""
        label_set = set()
        for f in self.files:
            sample = np.load(f, allow_pickle=True).item()
            label_set.add(sample["label"])
        return {label: idx for idx, label in enumerate(sorted(label_set))}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = np.load(self.files[idx], allow_pickle=True).item()
        x = sample["signal"]
        y = sample["label"]

        # Convert label if it's a string
        y = self.class_map[y] if isinstance(y, str) else y

        if self.transform:
            x = self.transform(x)
        else:
            x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        x = torch.tensor(x.copy(), dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def split_vibration_dataset(dataset, val_ratio=0.2, random_state=42):
    
    labels = np.array(dataset.labels)
    indices = np.arange(len(dataset))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(splitter.split(indices, labels))

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    return train_dataset, val_dataset
