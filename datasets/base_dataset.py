import os
import numpy as np
import librosa
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import download_file, extract_rar
from src.data_manager import DatasetManager

from abc import ABC, abstractmethod

class BaseDataset(ABC):
    
    def __init__(self, rawfilesdir, url):
        """
        Base class for all dataset models. 
        Defines the attributes and the download and load_signal functions, 
        along with an abstract extract_data method. The extract_data method 
        delegates the responsibility of data extraction to the subclasses, 
        requiring them to implement their specific extraction logic.

        Parameters:
        - rawfilesdir (str): The directory where raw files will be stored.
        - url (str): The base URL for downloading the dataset files.
        
        Methods:
            download(): Downloads .mat from the dataset website URL.
        """
        self._rawfilesdir = rawfilesdir  # Directory to store the raw files
        self._url = url  # Base URL for downloading the files
        self._data = [] # List to store the extracted data.
        self._label = []  # List to store the corresponding labels for the data.
        self.acquisition_maxsize = None  # Maximum size for data acquisition.
        dataset_name = str(self)
        self._annotation_file=DatasetManager(dataset_name).filter_data()
        self._metainfo = DatasetManager(dataset_name)
        self.map_classes = {"N": 0, "I": 1, "O": 2, "B": 3}
        self.all_signals = []
        self.all_labels = []
        self._is_cached = False
        self.dataset = None

        if not os.path.exists(self._rawfilesdir):
            os.makedirs(self._rawfilesdir)


    def download(self):
        """ Download files from datasets website.
        """
        url = self._url
        dirname = self.rawfilesdir
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        print(f"Stating download of {self} dataset.")
        list_of_bearings = self.list_of_bearings()
        dataset_name = self.__class__.__name__.lower()
        unit = '.mat'
        if dataset_name == "paderborn":
            unit = '.rar'
        for bearing in list_of_bearings:
            sufix_url = bearing[1]
            output_path = os.path.join('data/raw', dataset_name, bearing[0]+unit)
            if not os.path.exists(os.path.join(dirname, sufix_url)):
                download_file(url, sufix_url, output_path)                
            if unit == '.rar':
                extract_rar(output_path, output_path[:-4])
        print("Download finished.")

    # def load_signal_by_path(self, filepath):
    #     signal, label = self._extract_data(filepath)
    #     return signal, label
    
    # def load_signal(self, data_filter, segment_size=None, target_sr=None):
    #     if len(self.all_signals) != 0:
    #         return self.all_signals, self.all_labels
    #     metainfo = self._metainfo.filter_data(data_filter)
    #     signals = []
    #     labels = []
    #     if segment_size:
    #         for info in metainfo:
    #             basename = info["filename"]        
    #             filepath = os.path.join('data/raw/', self.__class__.__name__.lower(), basename+'.mat')            
    #             data, label = self.load_signal_by_path(filepath)
    #             if target_sr:
    #                 data = librosa.resample(data, orig_sr=self.sampling_rate, target_sr=target_sr)
    #             # split the signal into segments
    #             n_segments = data.shape[0] // segment_size
    #             for i in range(n_segments):
    #                 sample = data[(i * segment_size):((i + 1) * segment_size)]
    #                 signals.append(sample)
    #             labels_aquisition = np.full(n_segments, self.map_classes[label])
    #             labels.extend(labels_aquisition)
    #     else:
    #         for info in metainfo:
    #             basename = info["filename"]        
    #             filepath = os.path.join('data/raw/', self.__class__.__name__.lower(), basename+'.mat')            
    #             data, label = self.load_signal_by_path(filepath)
    #             if target_sr:
    #                 data = librosa.resample(data, orig_sr=self.sampling_rate, target_sr=target_sr)
    #             signals.append(data)
    #             labels.append(label)
    #     self.all_signals = np.array(signals)
    #     self.all_labels = np.array(labels)
    #     return self.all_signals, self.all_labels

    def save_cache(self, cache_filepath):
        print(' Saving cache')
        directory = cache_filepath.split('/')[0]
        os.makedirs(directory, exist_ok=True)
        with open(cache_filepath, 'wb') as f:
            np.save(f, self.all_signals)
            np.save(f, self.all_labels)
        self._is_cached = True
    
    def load_cache(self, cache_filepath):
        print(' Loading cache')
        with open(cache_filepath, 'rb') as f:
            self.all_signals = np.load(f)
            self.all_labels = np.load(f)
        return self.all_signals, self.all_labels
    
    def load_signal(self, data_filter, segment_size=None, target_sr=None, transform=None):
        """
        data_filter: Filtro para selecionar os arquivos de interesse.
        segment_size: Tamanho dos segmentos do sinal.
        target_sr: Taxa de amostragem desejada.
        transform: Transformação a ser aplicada nos sinais.
        """
        if self.dataset is not None:
            return self.dataset  # Retorna o dataset já carregado

        metainfo = self._metainfo.filter_data(data_filter)
        file_paths = []
        labels = []

        for info in metainfo:
            basename = info["filename"]
            filepath = os.path.join('data/raw/', self.__class__.__name__.lower(), basename + '.mat')
            file_paths.append(filepath)
            labels.append(self.map_classes[info["label"]])

        # Criando o dataset
        self.dataset = VibrationDataset(
            file_paths=file_paths,
            labels=labels,
            segment_size=segment_size,
            target_sr=target_sr,
            sampling_rate=self.sampling_rate,
            transform=transform
        )

        return self.dataset

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        """
        Retorna um DataLoader para treinar a rede neural.
        """
        if self.dataset is None:
            raise ValueError("Dataset ainda não foi carregado. Chame `load_signal` primeiro.")

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def get_metainfo(self, filter=None):
        return self._metainfo.filter_data(filter)

    @classmethod
    @abstractmethod
    def _extract_data(self, filepath):
        """ This method is responsible for extracting data from a bearing fault dataset in a .mat file.
        Returns:
            tuple: A tuple containing (data, label), where 'data' is the extracted dataset and 'label' is the corresponding label.
        """
        pass  

    @property
    def data(self):
        return self._data
    
    @property
    def label(self):
        return self._label
    
    @property
    def rawfilesdir(self):
        return self._rawfilesdir
        
    @property
    def annotation_file(self):
        return self._annotation_file
    
    @property
    def metainfo(self):
        return self._metainfo
    

class VibrationDataset(Dataset):
    def __init__(self, dataset, max_size=200_000, segment_size=None, target_sr=None, sampling_rate=None, transform=None, class_name=["N", "I", "O", "B"]):
        """
        file_paths: Lista de caminhos dos arquivos de dados (.mat).
        labels: Lista de rótulos correspondentes.
        segment_size: Tamanho do segmento para dividir o sinal.
        target_sr: Taxa de amostragem desejada (opcional).
        sampling_rate: Taxa de amostragem original do sinal.
        transform: Transformação a ser aplicada nos dados.
        """
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
