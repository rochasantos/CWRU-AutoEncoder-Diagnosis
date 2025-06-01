import os
import numpy as np
import torch
from torch.utils.data import Dataset

class VibrationDataset(Dataset):
    
    def __init__(self, data_dir, labels_map, selected_files=None):
        self.data_dir = data_dir
        all_files = os.listdir(data_dir)
        if selected_files:
            self.files = [f for f in all_files
                          if f.endswith('.npy') and any(f.startswith(prefix + "_") for prefix in selected_files)]
        else:
            self.files = [f for f in all_files if f.endswith('.npy')]
        self.labels_map = labels_map

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(file_path, allow_pickle=True).item()
        signal = data['signal'][:,0]
        label = data['label']    
        label = self.labels_map[label]
        label = torch.tensor(label, dtype=torch.long)
        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

        return signal, label
