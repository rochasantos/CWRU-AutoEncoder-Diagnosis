import os
import numpy as np
import torch
from torch.utils.data import Dataset

class VibrationDataset(Dataset):
    def __init__(self, data_dir, labels_map, selected_files=None, transform=None):
    
        self.data_dir = data_dir
        all_files = os.listdir(data_dir)
        self.labels_map = labels_map
        self.transform = transform

        # if selected_files:
        #     self.files = [f for f in all_files
        #                   if f.endswith('.npy') and any(f.startswith(prefix) for prefix in selected_files)]
        # else:
        self.files = [f for f in all_files if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(file_path, allow_pickle=True).item()
        signal = data['signal']
        label = data['label']
        # label = data['label'] if data['label'] == 'B' else 'D'        
        label = self.labels_map[label]

        if self.transform:
            signal = self.transform(signal)

        label = torch.tensor(label, dtype=torch.long)
        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

        return signal, label
