import os
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class VibrationDataset(Dataset):
    def __init__(self, dataset, group_info, sample_size, transform=None):        
        self.dataset = dataset
        self.ids = group_info[0]
        self.labels = group_info[1]
        self.start_position = group_info[2]
        self.sample_size = sample_size
        self.transform = transform
        self.file_list = self._load_file_paths(group_info[0])
        self.label_mapping = {
            "N": 0,  # Normal
            "I": 1,  # Inner Race Fault
            "O": 2,  # Outer Race Fault
            "B": 3,  # Ball Fault
            "H": 4   # Hybrid Fault (se aplicável)
        }

    def _load_file_paths(self, ids):        
        return [os.path.join(self.dataset.rawfilesdir, id+".mat") for id in ids]
    
    def load_file(self, filepath, start_position=0):
        signal, label = self.dataset.load_file(filepath)
        signal = signal[start_position:start_position+self.sample_size]
        return signal, label
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        signal, label = self.load_file(self.file_list[idx], int(self.start_position[idx]))
        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.label_mapping[label], dtype=torch.long)
        
        if self.transform:
            signal = self.transform(signal)
        
        return signal, label
