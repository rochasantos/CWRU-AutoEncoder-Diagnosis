import torch
import numpy as np
import os
from torch.utils.data import Dataset
from scipy.signal import hilbert


class VibrationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Diretório base onde estão as classes de falhas (I, O, B).
        transform: Transformações opcionais aplicadas ao sinal.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = self._load_file_paths()  # Carrega os caminhos dos arquivos

    def _load_file_paths(self):
        """
        Percorre os diretórios e carrega os caminhos de todos os arquivos .npy disponíveis.
        """
        file_paths = []
        for class_dir in os.listdir(self.root_dir):
            full_class_dir = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(full_class_dir):
                for filename in os.listdir(full_class_dir):
                    if filename.endswith(".npy"):
                        file_paths.append(os.path.join(full_class_dir, filename))
        return file_paths

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path)

        signal = data[:-1]  # Todos os valores menos o último (sinal)
        label = int(data[-1])  # Última posição é a etiqueta (já numérica)

        # 🔹 Aplica Transformada de Hilbert para criar o segundo canal
        hilbert_signal = np.imag(hilbert(signal))  # Parte imaginária do sinal analítico

        # 🔹 Empilha os dois canais: [original, hilbert_transform] → Formato: (2, num_samples)
        signal_2ch = np.stack([signal, hilbert_signal], axis=0)

        # 🔹 Converte para tensor PyTorch
        signal_2ch = torch.tensor(signal_2ch, dtype=torch.float32)  # (2, num_samples)
        label = torch.tensor(label, dtype=torch.long)

        return signal_2ch, label
