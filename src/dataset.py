from torch.utils.data import Dataset
import h5py
import torch

class VibrationDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path

    def __len__(self):
        with h5py.File(self.h5_path, "r") as f:
            return f["signals"].shape[0]

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            signal = f["signals"][idx]
            label = f["labels"][idx]

        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # Adiciona um canal extra

        return signal, torch.tensor(label, dtype=torch.long)
