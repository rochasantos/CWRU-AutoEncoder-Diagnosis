import numpy as np
import scipy.io
import os
import re
from datasets.base_dataset import BaseDataset

class Hust(BaseDataset):    
    """
    Hust Dataset Class

    This class manages the Hust bearing dataset used for fault diagnosis.
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

    def __init__(self, debug=False):
        
        super().__init__(rawfilesdir = "data/raw/hust", 
                         url = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/")
        self.sampling_rate = 51200
    def list_of_bearings(self):
        """ Returns a list of tuples with filenames (which will be used to name the downloaded files 
            in the destination directory) and URL suffixes to download vibration data.
        """
        return [
        ("B500", "9c521e5a-2554-48f9-b5f4-ea844efd0da3"),
        ("B502", "cc3c302e-eabd-4629-879c-65a23744fb9f"),
        ("B504", "76e85f66-d2ed-4ca7-a652-9bd63a22eef9"),
        ("B600", "9a38d892-4bf5-4154-96f0-4a5bece8cbb4"),
        ("B602", "0c4fc816-c9a7-46ec-9486-4b0b21edfd7a"),
        ("B604", "026c1b45-ea3d-46e5-ad7f-6eca8eded0b2"),
        ("B700", "01732e68-a452-4bd1-855c-eb58e1609596"),
        ("B702", "3632f984-c1a4-4c87-91b8-98171d3e49ea"),
        ("B704", "658e3634-4b2a-432a-9a2e-b8be12e6a2c5"),
        ("B800", "a02a8031-2a52-4118-ac35-437fab1fdc3c"),
        ("B802", "408d2ab7-912a-4c80-8a3b-ad19434b55da"),
        ("B804", "62b89baa-8ee0-4e63-bc7c-dfb09e2025f5"),
        ("I400", "16ddf73b-ee52-45e5-a08c-d678814ec09b"),
        ("I402", "bb4fce48-6f99-4ffa-b3a4-97c12c438872"),
        ("I404", "4426d37d-5a3d-4039-a245-a6becfc4674b"),
        ("I500", "9413a045-943a-473b-8590-27a860c8b4f5"),
        ("I502", "47045a1d-5320-4468-b9ab-b4041e4dae0b"),
        ("I504", "cdcaaaba-2614-43d5-b1c3-43c1a6657064"),
        ("I600", "233fad5e-314e-4e33-9e62-3b662d5993a8"),
        ("I602", "77587817-dda6-4aab-b398-5acf97041f09"),
        ("I604", "0d277d6a-7849-4757-ab94-26f22731fcfb"),
        ("I700", "5232dd7d-4d61-44af-bb34-425f97b9658e"),
        ("I702", "d9743b8a-599d-469d-9b70-e0ea3bea12d6"),
        ("I704", "c3f50f7d-c69c-4c20-b391-4eceb74715d5"),
        ("I800", "e1fb7270-e443-4cd1-8262-184fda765ee6"),
        ("I802", "7161a686-402a-459f-8a72-6174f0b708f9"),
        ("I804", "3be156ba-e7b3-4db8-a360-98c0c3a78a8e"),
        ("IB500", "b94c46e5-c822-4e37-8990-e6bbe013541e"),
        ("IB502", "6426959d-0927-42b8-a706-10db427afd0e"),
        ("IB504", "8b0576ff-7cae-477b-8f99-38998e55c35e"),
        ("IB600", "f63bca26-8749-4a70-8385-3b37b666b5d7"),
        ("IB602", "3650d5b3-f26a-49a2-ac3c-75ed2ef3a5ff"),
        ("IB604", "d59b696c-6bfb-42c4-b160-9051f3c8c585"),
        ("IB700", "469c0b32-952d-4588-9c8a-476d6b630ff7"),
        ("IB702", "0f40170e-3814-4db4-9730-58424bcc5ceb"),
        ("IB704", "bb487cb5-f304-4253-a3c2-3081a4797b0a"),
        ("IB800", "d821c83e-d887-45e7-903e-410940a7dc3b"),
        ("IB802", "e20768a2-10dc-4a56-b9a9-23a76ea77951"),
        ("IB804", "0177b728-f5b5-4f25-9911-d5baa6d1e5d7"),
        ("IO400", "c7c91200-024c-4b69-98e5-0963962042d3"),
        ("IO402", "34bf84e2-5bea-442e-81ad-8fed0624f206"),
        ("IO404", "0d8863ca-808a-4a10-a74d-635cd124af65"),
        ("IO500", "bcd6b2e1-891e-4cb1-a2a8-daf3d733eb89"),
        ("IO502", "2e491b9e-539a-4bc3-9321-16dfb204cea8"),
        ("IO504", "97637ad1-0a83-4c90-99ed-ead95ceff840"),
        ("IO600", "8106b110-cca7-465d-9c01-b31f8492e3fe"),
        ("IO602", "75ef1417-2830-44c4-97ab-bb9d2f6a2ee9"),
        ("IO604", "a94f2724-d8f4-41b3-914e-4ae8b68755d7"),
        ("IO700", "513c40f6-e328-4615-8087-85423c4e9b31"),
        ("IO702", "9e7fe437-3339-4500-abe5-a0d1bed3fbd3"),
        ("IO704", "ec06aaa8-c26e-4378-90f1-03dbd5cb8406"),
        ("IO800", "83214d5c-633b-4e02-a1e0-c2016d81893b"),
        ("IO802", "ed833b78-ba04-49f2-b258-b01e17f73c25"),
        ("IO804", "8ba809e0-b732-4f5f-a191-74c942024ca7"),
        ("N400", "17f0ca65-1bd5-4f04-9b1a-d7a81af46872"),
        ("N402", "035372ed-bf7c-4d7a-98ec-bad3e2cd3938"),
        ("N404", "ff6a8a30-e30b-4fd9-8d89-4a12b2dd3df6"),
        ("N500", "8efeb5d1-9b0a-4e2d-833c-51ac4ee86a38"),
        ("N502", "b7c7cb3e-fe3c-49e2-a156-8ccebb988eaa"),
        ("N504", "a1f28691-fe8e-4f4f-8fc2-496ac2a274b4"),
        ("N600", "cd58c6a6-d241-4ba9-b44d-8dc9df2d295b"),
        ("N602", "d19012cb-dc72-4bef-ae8e-a1882e5fa83b"),
        ("N604", "06eab7c4-9dc8-454e-9143-a0ab28864a7f"),
        ("N700", "eb68ac72-57ab-48f8-9ceb-dd868939bd50"),
        ("N702", "e50f2609-e18d-47e0-9424-e61dd83b61ee"),
        ("N704", "d1efe6a6-f88c-44f1-bbe3-0637bf354c0c"),
        ("N800", "21968130-1427-4e13-9d78-9ed365c50dd7"),
        ("N802", "f9690e72-3564-4d0b-89bf-1538b39a5de0"),
        ("N804", "4c9aaebc-db83-459f-a84a-b80e8bb4af3c"),
        ("O400", "58d3bed5-ca7a-49c8-bdf5-fcf9441cc94e"),
        ("O402", "54cdcf29-b5be-477c-ad09-49680bae3ba5"),
        ("O404", "aa1ecce9-8d06-47c3-af01-5da046f84030"),
        ("O500", "663046d5-3f87-47ca-a090-44a346f410ad"),
        ("O502", "3b3dc17e-9799-4e0f-bdf1-dac79280d743"),
        ("O504", "49650671-1d7c-486e-91e3-d2551aa22e75"),
        ("O600", "1cbc7965-66fb-424f-b0dc-a677292ae189"),
        ("O602", "fc14771a-3075-4df2-a12a-caf9a78aeae3"),
        ("O604", "53d36fe9-b71f-4019-b6ce-13c96331bcc0"),
        ("O700", "405bbef6-1b7d-4ec6-a0a2-7a1e6de05020"),
        ("O702", "85253e0d-ebd7-4575-b230-0c8a194a272f"),
        ("O704", "7695cdf6-0790-43cc-b6a5-c0e68744cd03"),
        ("O800", "a884e4a3-9c63-427d-b38f-eaae16d3b213"),
        ("O802", "e3fbd440-6f80-4595-92f3-677d2bd2dcb8"),
        ("O804", "bbaf54f7-b1b7-4e8a-a87c-403daaf8f5fc"),
        ("OB400", "9bde75e0-a421-40ac-941f-13195e5cf2b9"),
        ("OB402", "b1cf20bc-ff41-4a97-9f98-525089788eee"),
        ("OB404", "b2c42257-2a5d-4dd0-ad0d-5752b4165dc6"),
        ("OB500", "83125893-de12-4701-b43e-54444d453606"),
        ("OB502", "af62bb14-efc1-4dd9-982c-d2e5b937422c"),
        ("OB504", "5b10eda2-ce4c-44cf-b468-84012fc53d43"),
        ("OB600", "b96773e4-b778-496b-8b3d-cc901d9561c5"),
        ("OB602", "3d7f34e1-1bb6-4401-b7af-46720a99ec4d"),
        ("OB604", "45d96b9e-6e28-48b8-8d6b-0b6e59f7ed0a"),
        ("OB700", "9b138c7a-3c2a-42df-bc86-fce289406092"),
        ("OB702", "fc81258a-e19f-4ea7-9d13-a47691309f96"),
        ("OB704", "185ee762-b8bc-4fb1-83b5-2d4ad56e8f81"),
        ("OB800", "fbebb8aa-91d3-4ae3-9ca7-77a1655b3c8b"),
        ("OB802", "ac16f529-1f55-46b8-80fd-8842e3cbf597"),
        ("OB804", "c1d007a6-80ba-423a-ae9b-b1b63f4d7419")
        ]

    def _extract_data(self, filepath):
        """ Extracts data from a .mat file for bearing fault analysis.
        Args:
            filepath (str): The path to the .mat file.
        Return:
            tuple: A tuple containing the extracted data and its label.
        """
        matlab_file = scipy.io.loadmat(filepath)
        filename = os.path.basename(filepath)
        label = re.findall(r'[A-Z]{1,2}', filename)[0]       
        data_squeezed = np.squeeze(matlab_file["data"])
        if self.acquisition_maxsize:
            return data_squeezed[:self.acquisition_maxsize], label
        else:
            return data_squeezed, label
    
    def __str__(self):
        return "Hust"