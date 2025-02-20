import os
import numpy as np
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
    