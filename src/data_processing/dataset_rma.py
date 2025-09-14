# src/data_processing/vibration_dataset.py
import os
import re
import csv
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Dict, Optional, Sequence, Tuple, List

def _to_tensor_1d(x) -> torch.Tensor:
    """Ensure torch.float32 tensor with shape [1, L]."""
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.asarray(x))
    x = x.float()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x  # [1, L]

def _one_hot(idx: int, num_classes: int) -> torch.Tensor:
    t = torch.zeros(num_classes, dtype=torch.float32)
    if 0 <= idx < num_classes:
        t[idx] = 1.0
    return t

class VibrationDataset(Dataset):
    """
    Recursively finds .npy under root_dir and keeps only those whose parent
    directory name is one of the class labels. Compatible with attribute_mode
    ('none' | 'dirlevel' | 'regex' | 'csv').

    Returns (x, c_onehot, a_tensor) where:
      x: [1, L] float tensor
      c_onehot: [num_classes]
      a_tensor: [1] with 'a' normalized to [0,1]
    """
    def __init__(
        self,
        root_dir: str,
        classes: Sequence[str] = ("N", "I", "O", "B"),
        transform: Optional[Callable] = None,

        # Attribute config
        attribute_mode: str = "none",           # "none" | "dirlevel" | "regex" | "csv"
        dirlevel: int = 1,                      # used when attribute_mode="dirlevel"
        regex_pattern: Optional[str] = None,    # used when attribute_mode="regex"
        csv_path: Optional[str] = None,         # used when attribute_mode="csv"
        csv_attr_col: str = "a",                # column with attribute in CSV
        csv_relpath_col: str = "relpath",       # preferred; else use 'filepath'

        # Custom parser from token->float (optional)
        attr_token_to_float: Optional[Callable[[str], float]] = None,

        # Optional fixed normalization range
        attr_minmax: Optional[Tuple[float, float]] = None,
    ):
        self.root_dir = root_dir
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

        self.attribute_mode = attribute_mode
        self.dirlevel = dirlevel
        self.regex = re.compile(regex_pattern) if regex_pattern else None
        self.csv_path = csv_path
        self.csv_attr_col = csv_attr_col
        self.csv_relpath_col = csv_relpath_col
        self.attr_token_to_float = attr_token_to_float
        self.attr_minmax = attr_minmax

        # -------- Recursively collect samples --------
        # Pick every *.npy under root_dir, keep if parent folder is a class name
        self.samples: List[Tuple[str, int]] = []
        pattern = os.path.join(self.root_dir, "**", "*.npy")
        for path in glob.iglob(pattern, recursive=True):
            parent = os.path.basename(os.path.dirname(path))
            if parent in self.class_to_idx:
                self.samples.append((os.path.normpath(path), self.class_to_idx[parent]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No .npy files found under {self.root_dir} with classes {self.classes}")

        # -------- Optional CSV map for attributes --------
        self._csv_map: Dict[str, float] = {}
        if self.attribute_mode == "csv":
            if not self.csv_path or not os.path.isfile(self.csv_path):
                raise ValueError("attribute_mode='csv' requires a valid csv_path.")
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self.csv_relpath_col in row and row[self.csv_relpath_col]:
                        key = os.path.normpath(row[self.csv_relpath_col])
                    elif "filepath" in row and row["filepath"]:
                        key = os.path.normpath(row["filepath"])
                        if key.startswith(os.path.normpath(self.root_dir)):
                            key = os.path.relpath(key, self.root_dir)
                    else:
                        continue
                    try:
                        self._csv_map[key] = float(row[self.csv_attr_col])
                    except Exception:
                        continue
            if not self._csv_map:
                raise ValueError("CSV loaded but no rows mapped. Check column names and paths.")

        # -------- Compute attributes & normalization --------
        self._raw_attrs: List[float] = [self._extract_attr(path, lbl) for path, lbl in self.samples]
        if self.attribute_mode == "none":
            self._raw_attrs = [0.0 for _ in self.samples]

        if self.attr_minmax is None:
            a_min = float(min(self._raw_attrs)) if self._raw_attrs else 0.0
            a_max = float(max(self._raw_attrs)) if self._raw_attrs else 1.0
        else:
            a_min, a_max = self.attr_minmax
        self._a_minmax = (a_min, a_max)

        # Public aliases
        self.labels = [label for _, label in self.samples]
        self.targets = self.labels

    # ---------- Attribute helpers ----------

    def _token_to_float_default(self, token: str) -> float:
        tok = token.strip()
        if tok.isdigit() and len(tok) in (2, 3):
            return float(int(tok)) / 100.0
        try:
            return float(tok)
        except Exception:
            return 0.0

    def _normalize_attr(self, a: float) -> float:
        a_min, a_max = self._a_minmax
        if a_max <= a_min:
            return 0.0
        return float((a - a_min) / (a_max - a_min))

    def _extract_attr(self, path: str, label: int) -> float:
        if self.attribute_mode == "none":
            return 0.0

        rel = os.path.relpath(path, self.root_dir)  # works with recursion
        parts = rel.split(os.sep)

        if self.attribute_mode == "dirlevel":
            # Find index of the class dir in the path and step back 'dirlevel'
            try:
                class_dir_idx = max(i for i, p in enumerate(parts) if p in self.classes)
            except ValueError:
                class_dir_idx = len(parts) - 2
            attr_idx = class_dir_idx - self.dirlevel
            token = parts[attr_idx] if 0 <= attr_idx < len(parts) else ""
            parser = self.attr_token_to_float or self._token_to_float_default
            return parser(token)

        if self.attribute_mode == "regex":
            if self.regex is None:
                raise ValueError("attribute_mode='regex' requires regex_pattern.")
            m = self.regex.match(os.path.basename(path))
            token = m.group(1) if m else ""
            parser = self.attr_token_to_float or self._token_to_float_default
            return parser(token)

        if self.attribute_mode == "csv":
            rel_key = os.path.normpath(rel)
            if rel_key in self._csv_map:
                return float(self._csv_map[rel_key])
            abs_key = os.path.normpath(path)
            return float(self._csv_map.get(abs_key, 0.0))

        return 0.0

    # ---------- PyTorch Dataset API ----------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        x = np.load(path)  # expects 1D npy
        if self.transform is not None:
            x = self.transform(x)
        x = _to_tensor_1d(x)             # [1, L]

        c_onehot = _one_hot(label, len(self.classes))  # [C]
        a_raw = self._raw_attrs[idx]
        a_norm = self._normalize_attr(a_raw)
        a_tensor = torch.tensor([a_norm], dtype=torch.float32)  # shape [1]

        return x, c_onehot, a_tensor

