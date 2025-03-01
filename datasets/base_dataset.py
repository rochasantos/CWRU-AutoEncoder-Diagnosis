import os
import numpy as np
import librosa
from src.utils import download_file, extract_rar
from src.data_processing.data_manager import DatasetManager


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
        self.acquisition_maxsize = None  # Maximum size for data acquisition.
        self._metainfo = DatasetManager(str(self))
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

    def save_signal(self, root_dir="data/processed/cwru", data_filter=None, segment_size=None, target_sr=None, class_names=["I", "O", "B"]):        
        for cl in class_names:
            if root_dir.split("/")[-1] == 'cwru':
                for severity in ["007", "014", "021", "028"]:
                    if not os.path.exists(os.path.join(root_dir, cl)): 
                        os.makedirs(os.path.join(f"{root_dir}/{severity}", cl), exist_ok=True)
            elif not os.path.exists(os.path.join(root_dir, cl)): 
                os.makedirs(os.path.join(root_dir, cl), exist_ok=True)
        
        if isinstance(data_filter, dict):
            metainfo = self._metainfo.filter_data(data_filter)

        if segment_size:
            if root_dir.split("/")[-1] == 'cwru':
                for info in metainfo:
                    basename = info["filename"]        
                    filepath = os.path.join('data/raw/', self.__class__.__name__.lower(), basename+'.mat')            
                    signal, label = self._extract_data(filepath)
                    if target_sr:
                        signal = librosa.resample(signal, orig_sr=self.sampling_rate, target_sr=target_sr)
                    # split the signal into segments
                    n_segments = signal.shape[0] // segment_size
                    for i in range(n_segments):
                        sample = signal[(i * segment_size):((i + 1) * segment_size)]
                        label_value = np.array([class_names.index(label)])
                        data = np.hstack((sample, label_value))
                        np.save(f"{root_dir}/{info['extent_damage']}/{info['label']}/{basename}_{i}.npy", data)
            elif root_dir.split("/")[-1] == 'paderborn':
                for folder in data_filter:
                    lb = {'KA':'O', 'KI':'I'}[folder[:2]]
                    for file in os.listdir(os.path.join("data/raw/paderborn",folder, folder)):
                        if os.path.splitext(file)[1] != ".mat":
                            continue
                        filepath = os.path.join('data/raw/', self.__class__.__name__.lower(),folder, folder, file)            
                        signal, label = self._extract_data(filepath)
                        if target_sr:
                            signal = librosa.resample(signal, orig_sr=self.sampling_rate, target_sr=target_sr)
                        # split the signal into segments
                        n_segments = signal.shape[0] // segment_size
                        for i in range(n_segments):
                            sample = signal[(i * segment_size):((i + 1) * segment_size)]
                            label_value = np.array([class_names.index(label)])
                            data = np.hstack((sample, label_value))
                            np.save(f"{root_dir}/{lb}/{folder}_{i}.npy", data)
            else:
                for info in metainfo:
                    basename = info["filename"]        
                    filepath = os.path.join('data/raw/', self.__class__.__name__.lower(), basename+'.mat')            
                    signal, label = self._extract_data(filepath)
                    if target_sr:
                        signal = librosa.resample(signal, orig_sr=self.sampling_rate, target_sr=target_sr)
                    # split the signal into segments
                    n_segments = signal.shape[0] // segment_size
                    for i in range(n_segments):
                        sample = signal[(i * segment_size):((i + 1) * segment_size)]
                        label_value = np.array([class_names.index(label)])
                        data = np.hstack((sample, label_value))
                        np.save(f"{root_dir}/{info['label']}/{basename}_{i}.npy", data)

        else:
            for info in metainfo:
                basename = info["filename"]        
                filepath = os.path.join('data/raw/', self.__class__.__name__.lower(), basename+'.mat')            
                data, label = self._extract_data(filepath)
                if target_sr:
                    data = librosa.resample(data, orig_sr=self.sampling_rate, target_sr=target_sr)
                np.save(f"{root_dir}/{info['extent_damage']}/{info['label']}/{basename}.npy", np.array([data, label], dtype=object))

    
    def get_metainfo(self, filter=None):
        return self._metainfo.filter_data(filter)
    

    @classmethod
    @abstractmethod
    def _extract_data(self, filepath):        
        pass  

    @property
    def rawfilesdir(self):
        return self._rawfilesdir
        
    @property
    def metainfo(self):
        return self._metainfo
    