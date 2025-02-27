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

        # 🔹 Calcula o envelope do sinal com a Transformada de Hilbert
        envelope = np.abs(hilbert(signal))

        # 🔹 Aplica a Autocorrelação no envelope
        autocorr = np.correlate(envelope, envelope, mode='full')

        # 🔹 Mantém apenas a parte positiva da autocorrelação (metade superior)
        autocorr = autocorr[len(autocorr)//2:]

        # 🔹 Garante que autocorr tenha o mesmo tamanho do sinal original
        autocorr = autocorr[:len(signal)]

        # 🔹 Empilha os dois canais: [sinal original, autocorrelação do envelope]
        signal_2ch = np.stack([signal, autocorr], axis=0)  # Formato: (2, num_samples)

        # 🔹 Converte para Tensor PyTorch
        signal_2ch = torch.tensor(signal_2ch, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return signal_2ch, label
