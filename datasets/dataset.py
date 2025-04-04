import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PtDataset(Dataset):
    def __init__(self, root_dir, segment_size, transform=None):
        self.root_dir = root_dir       
        self.segment_size = segment_size
        self.transform = transform
        self.file_list = self._list_all_files(root_dir)
        self.label_mapping = {            
            "N": 0,  # Normal
            "I": 1,  # Inner Race Fault
            "O": 2,  # Outer Race Fault
            "B": 3,  # Ball Fault
            "H": 4   # Hybrid Fault (se aplicável)
        }

    def _list_all_files(self, root_dir):
        file_paths = []
        for root, _, files in os.walk(root_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                file_paths.append(full_path)
        file_paths_splited = []
        for path in file_paths:
            sample_size = np.load(path, allow_pickle=True).shape[0]-1
            for idx in range(sample_size//self.segment_size):
                file_paths_splited.append((path, idx*self.segment_size))
        print(f"Number of samples files: {len(file_paths_splited)}")
        return file_paths_splited
    
    def load_file(self, filepath, start_position=0):
        data = np.load(filepath, allow_pickle=True)
        signal, label = data[:-1], data[-1]
        signal = signal[start_position:start_position+self.segment_size]
        return signal, label
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        signal, label = self.load_file(self.file_list[idx][0], self.file_list[idx][1])
        
        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            signal = self.transform(signal)
        
        return signal, label
