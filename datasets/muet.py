import scipy.io
import os
from datasets.base_dataset import BaseDataset

class MUET(BaseDataset):    
    """
    UORED_VAFCLS Dataset Class

    This class manages the UORED_VAFCLS bearing dataset used for fault diagnosis.
    It provides methods for listing bearing files, loading vibration signals, and setting up dataset attributes.
    This class inherits from BaseDataset the load_signal methods responsible for loading and downloading data.
    
    Attributes
        rawfilesdir (str) : Directory where raw data files are stored.
        spectdir (str) : Directory where processed spectrograms will be saved.
        sample_rate (int) : Sampling rate of the vibration data.
        url (str) : URL for downloading the UORED-VAFCLS dataset.
        debug (bool) : If True, limits the number of files processed for faster testing.

    Methods
        list_of_bearings(): Returns a list of tuples with filenames and URL suffixes for downloading vibration data. 
        _extract_data(): Extracts the vibration signal data from .mat files.
        __str__(): Returns a string representation of the dataset.
    """
    
    def __init__(self):        
        super().__init__(rawfilesdir = "data/raw/uored",
                         url = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/")
        self.sampling_rate = 42000

    def list_of_bearings(self):
        """ 
        Returns: 
            A list of tuples containing filenames (for naming downloaded files) and URL suffixes 
            for downloading vibration data.
        """
        return [
        ("I_7_1", "2610f3a3-aeaa-4ac7-bb24-d614cc5ffb63"),
        ("I_7_2", "36ef0b5e-bfc0-4ad6-92b1-1c4d84b25101"),
        ("I_7_3", "3c3878cf-5dc3-455a-beb1-1bc34cb47e65"),
        ("O_7_1", "8db27e0a-eb56-49a1-b9bd-6e7913dab91a"),
        ("O_7_2", "6ce4bc79-c518-4c72-9a46-5e0fbd59a422"),
        ("O_7_3", "876786ca-a5e3-4f2e-b494-78165c157a21"),
        ("I_9_1", "1cc05f72-aa2c-469a-8a5d-27405aa6897f"),
        ("I_9_2", "a03ce899-e4a8-42f9-b560-e298543012cc"),
        ("I_9_3", "5875b84e-d754-4f3c-ae48-764087fdc188"),
        ("O_9_1", "bb323884-0e2a-48f7-bbdc-0cbd35a22ad0"),
        ("O_9_2", "03bba90f-a012-488b-8522-37ca6dd381f0"),
        ("O_9_3", "d9136d79-12ea-4225-b3cc-cfb61a9b5e90"),
        ("I_11_1", "84df1cda-a060-422d-ac1a-cbbc51534329"),
        ("I_11_2", "4789d55e-e2ba-49f8-b4ed-735cb20956b3"),
        ("I_11_3", "de4f61d6-c31a-4ee9-9c62-c4bbfc05f9b5"),
        ("O_11_1", "a47832ea-0ec1-4518-8300-e88cae4ae594"),
        ("O_11_2", "0d5262cb-95f6-4525-bd0a-3534f7fa0344"),
        ("O_11_3", "a5803a82-177f-4575-b7dd-3bd52cdca873"),
        ("I_13_1", "13154818-fbdf-4a93-bff7-457deda09ccc"),
        ("I_13_2", "7d322d93-280a-4a01-87e1-51f9e48ef23f"),
        ("I_13_3", "42a198e1-7d9f-462d-8a5f-c2cfe0988bc1"),
        ("O_13_1", "9db64a63-0e52-43ba-a239-ee3852d3c9e6"),
        ("O_13_2", "3e163890-7c5a-461e-8521-1e6829a015c3"),
        ("O_13_3", "61d1c398-a9bd-4cc7-9820-ee39d1360653"),
        ("I_15_1", "0c0136a0-8689-4779-8b79-bfb0c8e778ca"),
        ("I_15_2", "2ab2eb99-d8d1-4b35-b59c-cd7d7e96cccd"),
        ("I_15_3", "0fb376fa-d4ef-4e5f-b5ae-79671beea5ee"),
        ("O_15_1", "6dd2e6f9-f07b-4997-afcf-3d3d471bb4b3"),
        ("O_15_2", "5a60f1e2-51a3-4692-a6b1-077e89613dc4"),
        ("O_15_3", "ead67fdb-52c8-4d58-be82-6e5861fc3150"),
        ("I_17_1", "5ffe9031-070c-48b9-8f47-4a0b88f71b2c"),
        ("I_17_2", "98ebb2c8-8d3a-4066-bb9f-e039c257c31c"),
        ("I_17_3", "590d9582-5cbf-4199-ab29-147fcbc2b7a9"),
        ("O_17_1", "e0d6be9b-d1fa-4712-bae1-7d52eaf437ed"),
        ("O_17_2", "72f8306a-fdc0-45d0-88e1-6f718bbb3970"),
        ("O_17_3", "d7839b15-ce5a-4ac7-9951-f0d9b5dc99d5"),
        ("N_0_0", "4156c47e-a0fb-4dac-a545-2ff7fe5a00eb"),
        ("N_0_1", "a2567d61-560e-4c2e-8a44-49fd38df3dc1"),
        ]

    def _extract_data(self, filepath):
        """ Extracts data from a .mat file for bearing fault analysis.
        Args:
            filepath (str): The path to the .mat file.
        Return:
            tuple: A tuple containing the extracted data and its label.
        """
        
        matlab_file = scipy.io.loadmat(filepath)
        
        basename = os.path.basename(filepath).split('.')[0]
        file_info = list(filter(lambda x: x["filename"]==basename, self.get_metainfo()))[0]
        
        label = file_info["label"]
        data = matlab_file[basename][:, 0]
         
        if self.acquisition_maxsize:
            return data[:self.acquisition_maxsize], label
        else:
            return data, label

    def __str__(self):
        return "MUET"
    
if __name__ == "__main__":
    dataset = MUET()
    print(dataset)
