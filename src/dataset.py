import os
import librosa
import torch
from torch.utils.data import Dataset

class VibrationDataset(Dataset):

    def __init__(self, dataset, max_size=200_000, segment_size=None, target_sr=None, sampling_rate=None, transform=None, class_name=["N", "I", "O", "B"]):
        self.dataset = dataset
        self.class_name = class_name
        self.max_size = max_size
        self.segment_size = segment_size
        self.target_sr = target_sr
        self.sampling_rate = sampling_rate
        self.transform = transform

        self.file_paths = [
            os.path.join('data/raw/', self.dataset.__class__.__name__.lower(), info["filename"]+'.mat') 
            for info in self.dataset.get_metainfo()
            if self.dataset._extract_data(os.path.join('data/raw/', self.dataset.__class__.__name__.lower(), info["filename"]+'.mat'))[0].shape[0] >= self.max_size]
        print(f"The following files were uploaded:\n{[file.split('/')[-1] for file in self.file_paths]}")
        print(f"Totaling {len(self.file_paths)} files")
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Carrega o sinal do arquivo
        signal, label = self.dataset._extract_data(file_path)  # Ajuste conforme necessário
        signal = signal[:self.max_size]
        label = self.class_name.index(label)

        # Resampling, se necessário
        if self.target_sr and self.sampling_rate:
            signal = librosa.resample(signal, orig_sr=self.sampling_rate, target_sr=self.target_sr)

        # Segmentação do sinal
        if self.segment_size:
            n_segments = len(signal) // self.segment_size
            segments = [signal[i * self.segment_size:(i + 1) * self.segment_size] for i in range(n_segments)]
            labels = [label] * n_segments
        else:
            segments = [signal]
            labels = [label]

        # Aplicação de transformações (opcional)
        if self.transform:
            segments = [self.transform(torch.tensor(seg, dtype=torch.float32)) for seg in segments]
        
        return torch.stack(segments), torch.tensor(labels, dtype=torch.long)
