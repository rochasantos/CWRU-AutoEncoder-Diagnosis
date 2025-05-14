import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset


class VibrationMapBuilder:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.base_map = self._build_map()

    def _build_map(self):
        files = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith(".npy")
        ])
        map_list = []
        for idx, file in enumerate(files):
            sample = np.load(file, allow_pickle=True).item()
            label = sample["label"]
            map_list.append({"index": idx, "path": file, "label": label})
        return map_list

    def get_split(self, val_ratio=0.2, stratify=True):
        labels = [entry["label"] for entry in self.base_map]
        indices = np.arange(len(self.base_map))

        if stratify:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=None)
            train_idx, val_idx = next(splitter.split(indices, labels))
        else:
            from sklearn.model_selection import train_test_split
            train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=None, shuffle=True)

        train_map = [self.base_map[i] for i in train_idx]
        val_map = [self.base_map[i] for i in val_idx]
        return train_map, val_map




class VibrationDatasetFromMap(Dataset):
    def __init__(self, sample_map, transform=None, class_map=None):
        self.sample_map = sample_map
        self.transform = transform
        self.class_map = class_map or self._build_class_map()

    def _build_class_map(self):
        unique = sorted(set(entry["label"] for entry in self.sample_map))
        return {label: idx for idx, label in enumerate(unique)}

    def __len__(self):
        return len(self.sample_map)

    def __getitem__(self, idx):
        entry = self.sample_map[idx]
        sample = np.load(entry["path"], allow_pickle=True).item()
        signal = sample["signal"]
        label = self.class_map[sample["label"]]

        if self.transform:
            signal = self.transform(signal)
        else:
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        x = torch.tensor(signal.copy(), dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
