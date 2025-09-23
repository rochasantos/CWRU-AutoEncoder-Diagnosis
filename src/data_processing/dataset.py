import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset

# Label maps
LABELS = {
    "bin": {"N": 0, "F": 1},                # non-fault vs fault
    "bio": {"B": 0, "I": 1, "O": 2},        # only fault types
    "bino": {"N": 0, "B": 1, "I": 2, "O": 3} # all classes together
}

class BearingDataset(Dataset):
    """
    Loads .npy 1D signals organized as:
      root/bin/(N_1_0.npy,...,F_2_1.npy)   -> task="bin"
      root/bio/(B_5_2.npy,...,I_9_5.npy,...) -> task="bio"
      root/bino/(N_1_0.npy,B_2_1.npy,...) -> task="bino"
    Label is inferred from the first char in filename.
    """

    def __init__(self, root_dir, task="bin", transform=None, target_transform=None, return_path=False):
        self.root_dir = root_dir
        self.task = task.lower().strip()
        if self.task not in LABELS:
            raise ValueError("task must be 'bin', 'bio' or 'bino'")
        self.map = LABELS[self.task]
        self.transform = transform
        self.target_transform = target_transform
        self.return_path = return_path

        # collect files
        paths = sorted(glob(os.path.join(root_dir, "*.npy")))
        if not paths:
            raise FileNotFoundError("No .npy files found in %s" % root_dir)

        # keep only files whose first char is in label map
        self.samples = [p for p in paths if os.path.basename(p)[0] in self.map]
        if not self.samples:
            raise FileNotFoundError("No valid labeled files (by prefix) in %s" % root_dir)

        # store class names in index order
        inv = sorted(self.map.items(), key=lambda x: x[1])
        self.classes = [k for k, _ in inv]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        x = np.load(path)                 # expects 1D numpy array
        x = torch.from_numpy(x).float()   # to float32 tensor
        if x.ndim == 1:
            x = x.unsqueeze(0)            # shape [1, L] for Conv1d

        y_char = os.path.basename(path)[0]
        y = torch.tensor(self.map[y_char], dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        if self.return_path:
            return x, y, path
        return x, y



LABELS = {
    "bin":  {"N": 0, "F": 1},
    "bio":  {"B": 0, "I": 1, "O": 2},
    "bino": {"N": 0, "B": 1, "I": 2, "O": 3},
    "n_bio": {"N": 0, "B": 1, "I": 1, "O": 1},
    "b_ino": {"B": 1, "I": 0, "N": 0, "O": 0},
    "i_bno": {"I": 1, "B": 0, "N": 0, "O": 0},
    "o_bin": {"O": 1, "B": 0, "I": 0, "N": 0},
}

CLASS_NAMES = {
    "bin":  ["N", "F"],
    "bio":  ["B", "I", "O"],
    "bino": ["N", "B", "I", "O"],
    "n_bio": ["N", "BIO"],
    "b_ino": ["INO", "B"],
    "i_bno": ["BNO", "I"],
    "o_bin": ["BIN", "O"],
}

class BearingNPYDataset(Dataset):
    def __init__(self, root_dir, task="bin", transform=None, target_transform=None, return_path=False):
        self.root_dir = root_dir
        self.task = task.lower().strip()
        if self.task not in LABELS:
            raise ValueError("task must be one of: " + ", ".join(LABELS.keys()))
        self.map = LABELS[self.task]
        self.transform = transform
        self.target_transform = target_transform
        self.return_path = return_path

        paths = sorted(glob(os.path.join(root_dir, "*.npy")))
        if not paths:
            raise FileNotFoundError("No .npy files found in %s" % root_dir)

        allowed = set(self.map.keys())
        self.samples = [p for p in paths if os.path.basename(p)[0] in allowed]
        if not self.samples:
            raise FileNotFoundError("No valid labeled files (by prefix) for task '%s' in %s" % (self.task, root_dir))

        # NEW: expose numeric targets for samplers
        self.targets = [self.map[os.path.basename(p)[0]] for p in self.samples]

        self.classes = CLASS_NAMES[self.task]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        x = np.load(path)
        x = torch.from_numpy(x).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [1, L]

        # Use precomputed target
        y = torch.tensor(self.targets[idx], dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        if self.return_path:
            return x, y, path
        return x, y
