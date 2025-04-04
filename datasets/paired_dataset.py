import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict


class PairedPtDataset(Dataset):
    def __init__(self, root_dir, segment_size, transform=None):
        self.root_dir = root_dir       
        self.segment_size = segment_size
        self.transform = transform
        self.label_mapping = {            
            "N": 0,  # Normal
            "I": 1,  # Inner Race Fault
            "O": 2,  # Outer Race Fault
            "B": 3,  # Ball Fault
            "H": 4   # Hybrid Fault (se aplicável)
        }
        self.samples_by_class = self._index_all_files(root_dir)

    def _index_all_files(self, root_dir):
        """Indexa os segmentos disponíveis agrupando por classe"""
        class_dict = defaultdict(list)
        for root, _, files in os.walk(root_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                data = np.load(full_path, allow_pickle=True)
                label = int(data[-1])
                sample_size = data.shape[0] - 1
                for idx in range(sample_size // self.segment_size):
                    start_pos = idx * self.segment_size
                    class_dict[label].append((full_path, start_pos))
        print(f"Total de amostras indexadas: {sum(len(v) for v in class_dict.values())}")
        return class_dict

    def __len__(self):
        # Usa a menor classe para evitar overindex
        return min(len(v) for v in self.samples_by_class.values())

    def load_segment(self, path, start_pos):
        data = np.load(path, allow_pickle=True)
        signal = data[start_pos:start_pos + self.segment_size]
        return torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # [1, segment_size]

    def __getitem__(self, idx):
        # Escolhe aleatoriamente uma classe
        selected_class = random.choice(list(self.samples_by_class.keys()))
        candidates = self.samples_by_class[selected_class]

        # Garante que pegaremos duas amostras de arquivos diferentes
        pair = random.sample(candidates, 2)

        x1 = self.load_segment(*pair[0])
        x2 = self.load_segment(*pair[1])

        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        return x1, x2  # Par da mesma classe, aquisições diferentes
