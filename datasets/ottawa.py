import scipy.io
import os
from datasets.base_dataset import BaseDataset

class UORED(BaseDataset):    
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
        ("H_1_0", "31863372-55f7-4c9c-91a5-4f3c907a85af"),
        ("H_2_0", "7615c4b8-7c8f-41fd-8034-5b6e8438eb16"),
        ("H_3_0", "e5abb9af-727e-4fd5-8238-0d007f4be6d6"),
        ("H_4_0", "371e6895-8ae8-4e30-9925-0aacb815dfe9"),
        ("H_5_0", "ac2a3654-a2ae-4d99-a6a8-9b8e3f7eab27"),
        ("H_6_0", "960e11a3-e88b-40a2-8d05-e5051951599d"),
        ("H_7_0", "75bf8b30-beaf-4267-9190-db1ee68c3f14"),
        ("H_8_0", "0714450f-ebbb-4039-b415-60fc4bef3601"),
        ("H_9_0", "9cd35272-6b46-4fb5-b482-05ab53cf1c95"),
        ("H_10_0", "128dde70-6511-4404-9779-28d4a9da7438"),
        ("H_11_0", "8f7cc1ad-04eb-4b68-99ef-d2adb9ed8825"),
        ("H_12_0", "2ca4ba4a-146c-4d56-a309-889888713491"),
        ("H_13_0", "c1cd9d5d-61e2-468b-bf25-9ddeeff4b487"),
        ("H_14_0", "30fa3b7b-db98-4328-be88-1317ebedbbb6"),
        ("H_15_0", "75552a72-ed20-4723-8217-afa86e341099"),
        ("H_16_0", "4e4b52bb-f657-4347-8965-c77b7342f905"),
        ("H_17_0", "edecdcaf-c6eb-4125-bb63-2b704f33956f"),
        ("H_18_0", "3768c855-d514-42b3-ae11-700282d57089"),
        ("H_19_0", "2c74bcb6-d879-4a9a-863c-b257c91a5dc0"),
        ("H_20_0", "45ab549d-075f-4409-b76e-b8443fa0ca0b"),
        ("I_1_1", "d383ee8d-75ae-49bd-9fb5-6148695ab69b"),
        ("I_1_2", "fb1cca43-124d-4b74-b944-f0abc3f28132"),
        ("I_2_1", "25e8e452-7bee-40ae-9d8a-63f32c3325a4"),
        ("I_2_2", "0f7addcb-ff4d-4f15-9b91-5493b5bc0dcf"),
        ("I_3_1", "9c815a92-e72d-48a3-8bc9-14a51f6483fa"),
        ("I_3_2", "05922ac8-52cd-4189-b23c-a8bc5c9b88eb"),
        ("I_4_1", "883ba133-1610-4a7c-8084-bb0e941125e2"),
        ("I_4_2", "113815c9-ddfe-4f63-8f0e-76c75b6c0581"),
        ("I_5_1", "770d3546-6895-4c99-bfb4-7e57b1d7228d"),
        ("I_5_2", "c5415d06-2607-4878-b6f5-06cc67b30540"),
        ("O_6_1", "b3b96601-3e54-4634-95b3-6dd8aeddf613"),
        ("O_6_2", "c280f3a7-90e6-4438-9ce2-db7b84eb097c"),
        ("O_7_1", "805c124e-df28-40a5-933e-2e1976fbb73a"),
        ("O_7_2", "eeb7d7a3-87f3-468a-9038-6224cb256a0a"),
        ("O_8_1", "66834062-c2d5-45bd-9a72-8960828a8902"),
        ("O_8_2", "0500d6e7-02fa-4fcb-8df6-96ade544e3e4"),
        ("O_9_1", "ea795582-032f-4cbd-9601-5bce5873d1f8"),
        ("O_9_2", "4a8bfeb4-c031-450d-86cc-78e9e7cec17d"),
        ("O_10_1", "e3b6e7c5-f4c9-492c-ba14-7da2f1955709"),
        ("O_10_2", "798664fc-ce4f-4d0c-99e0-a30bc08cd6dc"),
        ("B_11_1", "8d34facf-e922-46cb-9ee7-f04edf7a89d2"),
        ("B_11_2", "0c1ad1e3-196e-466d-968a-41803f2d020e"),
        ("B_12_1", "7b9f2066-b6b5-4b0e-8fce-30f0bd15574f"),
        ("B_12_2", "8837b687-1f08-4cb7-9c98-1bb1823723b4"),
        ("B_13_1", "a0a2c68c-478b-495d-868c-ffae62b5b445"),
        ("B_13_2", "a3bdcff7-b9d4-417b-b902-b5cfd08e0985"),
        ("B_14_1", "7320c429-b6dc-4d4c-9dc4-57df635e603e"),
        ("B_14_2", "1e5409ec-d96c-4d02-9ef6-69bfd3d6ff2a"),
        ("B_15_1", "03de7b86-b6e5-4e1b-8e40-d7f9fbed8a12"),
        ("B_15_2", "17cbbffb-987a-4706-b1b4-778012b3ac16"),
        ("C_16_1", "ce267f69-4027-4db6-9f23-81c96029b523"),
        ("C_16_2", "12e2565a-99da-41df-ae49-70f1f5b2fddf"),
        ("C_17_1", "59b8814c-5a04-40fe-bfd2-7e712414d68f"),
        ("C_17_2", "1849dd9a-55b0-4313-90d8-fd91e3f3fde6"),
        ("C_18_1", "c2167e9e-e981-4ca2-ac7c-7dc31e058795"),
        ("C_18_2", "2654cd65-e54c-4d88-9554-9dbf56a6c60e"),
        ("C_19_1", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("C_19_2", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("C_20_1", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("C_20_2", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
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
        return "UORED"
    
if __name__ == "__main__":
    dataset = UORED()
    print(dataset)
