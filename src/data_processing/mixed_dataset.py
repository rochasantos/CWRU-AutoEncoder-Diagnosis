import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class MixedSignalDataset(Dataset):
    def __init__(self, root_dir, num_samples, overlap_fn, transform=None, segment_len=2048, seed=None):
        """
        Args:
            root_dir (str): Path to directory containing .npy files.
            num_samples (int): Number of mixed samples to generate per epoch (__len__).
            overlap_fn (callable): Function that receives two numpy 1D arrays and returns a mixed 1D array.
            transform (callable, optional): Optional transform applied to the mixed tensor (torch).
            segment_len (int): Segment length to slice from each source signal.
            seed (int, optional): RNG seed for reproducibility of sampling.
        """
        self.root_dir = root_dir
        self.num_samples = int(num_samples)
        self.overlap_fn = overlap_fn
        self.transform = transform
        self.segment_len = int(segment_len)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Acquisition ranges known for each class prefix
        self.aquisitions_opts = {
            'N': list(range(1, 11)),   # H nos seus .mat, mas aqui você disse .npy com prefixo 'N'
            'I': list(range(1, 6)),
            'O': list(range(6, 11)),
            'B': list(range(11, 16))
        }

        # List all .npy files present
        self.samples = sorted([f for f in os.listdir(root_dir) if f.endswith(".npy")])

        if not self.samples:
            raise RuntimeError(f"No .npy files found under {root_dir}")

        # Build classes actually present (intersection with known prefixes)
        present_prefixes = sorted(set(f.split("_")[0] for f in self.samples))
        self.classes = [c for c in ['N', 'I', 'O', 'B'] if c in present_prefixes]
        if not self.classes:
            raise RuntimeError("No valid class prefixes found among files (expected one of N/I/O/B).")

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Pre-index files by (class, acquisition) for fast pairing
        # Expect filenames like "<C>_<acq>_<...>.npy", e.g., "B_11_*.npy"
        self.by_class_acq = {}
        for f in self.samples:
            parts = f.split("_")
            if len(parts) < 3:
                # Not in expected pattern; skip
                continue
            cls = parts[0]
            if cls not in self.class_to_idx:
                continue
            try:
                acq = int(parts[1])
            except ValueError:
                continue
            self.by_class_acq.setdefault(cls, {}).setdefault(acq, []).append(f)

        # Targets aligned to samples (for samplers etc.)
        self.targets = []
        for f in self.samples:
            cls = f.split("_")[0]
            if cls in self.class_to_idx:
                self.targets.append(self.class_to_idx[cls])
            else:
                # Put -1 for unknown; these files won't be chosen in __getitem__ if they lack a valid pair
                self.targets.append(-1)

        # Quick sanity: ensure there is at least one class with ≥2 acquisitions and files
        ok_any = False
        for cls in self.classes:
            acqs = [a for a in self.aquisitions_opts.get(cls, []) if a in self.by_class_acq.get(cls, {})]
            if len(acqs) >= 2:
                ok_any = True
                break
        if not ok_any:
            raise RuntimeError("No class has at least two acquisitions with files to form a pair.")

    @property
    def num_classes(self):
        return len(self.classes)

    def __len__(self):
        # Make the DataLoader call __getitem__ exactly num_samples times per epoch
        return self.num_samples

    def _pick_pair_paths(self):
        """
        Pick two file paths of the same class but from different acquisitions.
        Returns (path1, path2, cls_str).
        """
        # Pick a random class that has at least 2 acquisitions available
        valid_classes = []
        for cls in self.classes:
            acqs_available = [a for a in self.aquisitions_opts.get(cls, []) if a in self.by_class_acq.get(cls, {})]
            if len(acqs_available) >= 2:
                valid_classes.append(cls)
        if not valid_classes:
            raise RuntimeError("No valid class with two different acquisitions available.")

        cls = random.choice(valid_classes)
        acqs = [a for a in self.aquisitions_opts[cls] if a in self.by_class_acq[cls]]
        acq1 = random.choice(acqs)
        acqs2 = [a for a in acqs if a != acq1]
        acq2 = random.choice(acqs2)

        # Pick one file from each acquisition
        f1 = random.choice(self.by_class_acq[cls][acq1])
        f2 = random.choice(self.by_class_acq[cls][acq2])

        path1 = os.path.join(self.root_dir, f1)
        path2 = os.path.join(self.root_dir, f2)
        return path1, path2, cls

    def __getitem__(self, idx):
        # Finite stop for manual for-loops over the dataset
        if idx >= self.num_samples:
            raise IndexError

        # Choose two signals (same class, different acquisitions)
        path1, path2, cls = self._pick_pair_paths()

        sig1 = np.load(path1).squeeze()
        sig2 = np.load(path2).squeeze()

        if sig1.ndim != 1:
            sig1 = sig1.reshape(-1)
        if sig2.ndim != 1:
            sig2 = sig2.reshape(-1)

        L1, L2 = sig1.shape[0], sig2.shape[0]
        L = self.segment_len
        if L1 < L or L2 < L:
            raise ValueError(f"Signal too short for segment_len={L}: {path1}({L1}), {path2}({L2})")

        # Random crops
        i = random.randint(0, L1 - L)
        j = random.randint(0, L2 - L)
        x1 = sig1[i:i+L]
        x2 = sig2[j:j+L]

        # User-provided mixing (expects numpy arrays)
        mixed = self.overlap_fn(x1, x2)

        # To torch
        if not isinstance(mixed, torch.Tensor):
            mixed = torch.from_numpy(np.asarray(mixed))
        mixed = mixed.float()
        if mixed.ndim == 1:
            mixed = mixed.unsqueeze(0)
        # mixed = torch.as_tensor(np.asarray(mixed), dtype=torch.float32)

        if self.transform is not None:
            mixed = self.transform(mixed)

        # Label tensor (long)
        label_idx = self.class_to_idx[cls]
        label = torch.tensor(label_idx).long()

        return mixed, label
