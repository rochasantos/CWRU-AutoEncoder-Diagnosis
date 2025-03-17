import scipy.io
import os
from datasets.base_dataset import BaseDataset

class SU(BaseDataset):    
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
        super().__init__(rawfilesdir = "data/raw/su",
                         url = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/")

    def list_of_bearings(self):
        """ 
        Returns: 
            A list of tuples containing filenames (for naming downloaded files) and URL suffixes 
            for downloading vibration data.
        """
        return [
        ("H_B_16_6204_1000", "86818d0a-4c33-4b48-8f6d-5f7318dcde3d"),
        ("H_H_16_6204_1000", "c2db78a0-8794-4de8-8549-79346dabb54d"),
        ("H_IR_16_6204_1000", "e8d85ff5-7d0b-4072-a787-448731f8c429"),
        ("H_OR_16_6204_1000", "dbb6d14a-3638-467b-9b8b-e688a5438294"),
        ("L_B_16_6204_1000", "e7a8b84b-0469-4c64-b1fc-250b98d0e2a6"),
        ("L_H_16_6204_1000", "bcb75c81-0828-4ce6-8989-33e7c16a95a6"),
        ("L_IR_16_6204_1000", "5c7e7b74-75fd-440c-8c2c-1072730f7212"),
        ("L_OR_16_6204_1000", "07626692-eb59-4bda-a209-af87b8c696c4"),
        ("M1_B_16_6204_1000", "72f2c8bb-fde5-4fe6-af12-85c947125bb2"),
        ("M1_H_16_6204_1000", "736b6c1a-3353-46ae-b182-47e0f2e4b651"),
        ("M1_IR_16_6204_1000", "b8e20a32-1e69-4e79-b87c-064a922a1372"),
        ("M1_OR_16_6204_1000", "3f407eb7-8c4d-40ce-a370-eb8e4c11f30b"),
        ("M2_B_16_6204_1000", "8ac6a56d-56ed-4305-a777-aba169dc73cf"),
        ("M2_H_16_6204_1000", "95551bf5-3bc2-4a41-a4a4-d325867fe358"),
        ("M2_IR_16_6204_1000", "a3d98df8-4594-49d6-b992-125485aa11e3"),
        ("M2_OR_16_6204_1000", "2c046667-1188-490c-b8f7-393ce6fd7da7"),
        ("M3_B_16_6204_1000", "9b9ac00e-1ae4-44c4-bb1e-78aa445ad3e0"),
        ("M3_H_16_6204_1000", "f3d5c1fc-a4ce-4c02-a7df-8f83bea9fa4e"),
        ("M3_IR_16_6204_1000", "0193ba38-a063-4596-9f24-d70d4ac1dd8f"),
        ("M3_OR_16_6204_1000", "7003bbfc-8b57-4437-93c5-8d485fa15f31"),
        ("U1_B_16_6204_1000", "ed43cab8-28ca-47ab-887c-2697b63a4699"),
        ("U1_H_16_6204_1000", "fb3d7f8f-7eda-4e9d-bf33-8cc6ae2ce433"),
        ("U1_IR_16_6204_1000", "2358e2a6-631f-461c-91cb-925d37cd4f1c"),
        ("U1_OR_16_6204_1000", "dd669063-59eb-4214-a064-f5b3a5135743"),
        ("U2_B_16_6204_1000", "93f4fd34-6935-4dda-a8ce-e07c2a02a1e3"),
        ("U2_H_16_6204_1000", "01053d81-2cfd-4f41-ae87-3e7043a9f5d7"),
        ("U2_IR_16_6204_1000", "837f02ed-af57-4ecc-af46-43f31c28eb0a"),
        ("U2_OR_16_6204_1000", "07f3dbdd-b69f-4426-876f-82f2ed57e257"),
        ("U3_B_16_6204_1000", "d2a721ef-22ce-490d-a66a-e2c3294d52a5"),
        ("U3_H_16_6204_1000", "efca97e1-1cbc-4f7d-a319-fda0e573aead"),
        ("U3_IR_16_6204_1000", "36ebf037-472f-434f-91bf-f89efab53145"),
        ("U3_OR_16_6204_1000", "09b6424d-605c-43b4-916b-22f68962cbbd"),
        ("H_B_16_6204_1200", "44b8ba13-9c64-428c-9118-ffb323a3063d"),
        ("H_H_16_6204_1200", "d10c2a02-6a85-46b9-80ad-b3d416d6f218"),
        ("H_IR_16_6204_1200", "979e2683-8d62-4421-af91-c474de9da997"),
        ("H_OR_16_6204_1200", "75aa2f63-4075-4053-bb8a-561bf78b0373"),
        ("L_B_16_6204_1200", "59305c97-bd51-4afd-8b92-84d21f4d2ac5"),
        ("L_H_16_6204_1200", "d85927bc-cb0d-4151-9a6a-46be29393f6e"),
        ("L_IR_16_6204_1200", "2a2627e6-340b-4f35-953a-1e350a39a019"),
        ("L_OR_16_6204_1200", "59305c97-bd51-4afd-8b92-84d21f4d2ac5"),
        ("M1_B_16_6204_1200", "06c258ad-1e9f-400e-89c3-115e0a2572ba"), 
        ("M1_H_16_6204_1200", "de622058-bc6b-49fa-b5c2-5179921e37b9"),
        ("M1_IR_16_6204_1200", "b78bc513-8521-496c-b41e-03784b2aac88"),
        ("M1_OR_16_6204_1200", "69186e54-b840-49ad-a04d-0de269ea203f"),
        ("M2_B_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/2122d2dc-349e-4ebd-9edd-be57f079c9be"),
        ("M2_H_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/c010d5ef-e4cf-475e-b836-387b5a5156cc"),
        ("M2_IR_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/557cdbab-1930-4917-97fa-d24fce725bbe"),
        ("M2_OR_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/3c0bfabb-a48b-487e-8aa1-a4a446d5d820"),
        ("M3_B_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/2f98cb52-5d47-41ff-b0df-00587800b7ec"),
        ("M3_H_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/c8df65cb-d344-4eb8-9407-e0583798724e"),
        ("M3_IR_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/9e5cb7ec-ec8c-465d-9351-813452eb969b"),
        ("M3_OR_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/9215814f-efc3-4de3-90d7-95e234decd16"),
        ("U1_B_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/a4581e80-13fd-4c1c-9785-f9205ae21636"),
        ("U1_H_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/bc68ab7c-fc5b-4872-9c47-65680f89defa"),
        ("U1_IR_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/5493141c-d504-4cf9-9dad-dba81339d86f"),
        ("U1_OR_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/868693a4-78da-40cf-ab3c-e26e69b0d7bb"),
        ("U2_B_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/35a1a832-3de2-4346-9760-0217d6f4c589"),
        ("U2_H_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/bd0585f0-3eaa-4142-8159-6327785aaa80"),
        ("U2_IR_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/7c3efa04-5c2a-4bcc-a7e1-30bc33b17e7e"),
        ("U2_OR_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/8be38556-4919-429c-a8f9-04f8dd8de3d4"),
        ("U3_B_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/a75947c6-462c-497b-87a2-0d23d8cfcc31"),
        ("U3_H_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/63cac031-b4a7-44c7-8cbb-747c89e1906c"),
        ("U3_IR_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/d09b1713-0f21-4203-8948-0cea9864eb34"),
        ("U3_OR_16_6204_1200", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/c988a856-894b-466b-9285-62e88a96be09"),
        ("H_B_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/d9eefa63-3d20-4ea4-b3df-32ff37989752"),
        ("H_H_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/acf09a0c-ed43-4c80-850e-cf65546459be"),
        ("H_IR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/db0fcf26-cbd3-4dfd-ad26-c99dd57db84d"),
        ("H_OR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/041339a1-0dea-4434-bb79-b78efb62efba"),
        ("L_B_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/e82d2f48-9a62-45b6-be3d-de1567ac58d3"),
        ("L_H_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/0ce5b757-13e7-4e5c-b759-8572623c120a"),
        ("L_IR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/53537126-c660-4c5b-8080-1d0e12eb1b10"),
        ("L_OR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/1c2c26dc-f5ee-4e5f-a86c-9664cb487668"),
        ("M1_B_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/bf85782f-f276-419a-a0bc-3052b14b2400"),
        ("M1_H_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/d811a745-f83b-456e-bc11-5185e1b467a6"),
        ("M1_IR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/2d51b33c-7ef5-4aae-88c2-e32363ef17fd"),
        ("M1_OR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/0a006ed9-123b-4b11-95d0-38c382ffe9e3"),
        ("M2_B_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/eedea089-4f59-4818-a42f-b19ebed3135c"),
        ("M2_H_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/d8c71eee-5c9e-46bf-ac04-bf404a47e40f"),
        ("M2_IR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/0b2ac6c3-f4e1-459f-840e-ddbf9a491670"),
        ("M2_OR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/a308ef87-d2f0-4fc2-be9c-a1a9d85d56a9"),
        ("M3_B_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/434d1873-6234-4a5c-b77e-27e5cfa14c69"),
        ("M3_H_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/b94afa4c-1e5d-4030-bc34-776ca10581b8"),
        ("M3_IR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/581b5fb5-1bf9-41d5-b1d3-8b5e561488ff"),
        ("M3_OR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/8b025537-e3f5-4063-b3c6-df107e661167"),
        ("U1_B_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/2cb335f8-144c-42b9-8fbc-5707a7719509"),
        ("U1_H_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/a4dbb3db-9523-468b-a99e-229f38b89d52"),
        ("U1_IR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/790ccf3c-a2ca-4b88-b5ae-95dd37a316fb"),
        ("U1_OR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/0adb3ce2-e40e-45c7-b5c4-dc30aa5c8dd0"),
        ("U2_B_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/fa7f4159-8291-447d-a7c0-9b998ed513ea"),
        ("U2_H_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/3d39dd9b-4e5f-4f6c-8e7b-67e378434799"),
        ("U2_IR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/0f381a3a-d2cc-4e6b-ba60-ad36c98c7395"),
        ("U2_OR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/1386ae80-363e-43d7-bfed-39bcbddde696"),
        ("U3_B_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/4ef27cc1-7e7d-4cfa-8faf-bbc885eac179"),
        ("U3_H_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/b83dea15-9dbd-41a2-aacc-b7d195a27027"),
        ("U3_IR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/68383fa8-da64-45fc-bc90-6363489b1cdf"),
        ("U3_OR_16_6204_1400", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/9c35a5f2-3877-41e9-8195-950e5e803bff"),
        ("H_B_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/f4c148d3-3eb9-44f4-bbb0-fafd03bf106b"),
        ("H_H_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/c0a24d70-6f3c-46ac-b1f8-94214229c20d"),
        ("H_IR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/6d67ab70-b08a-4961-b3ae-5a54ff3b80ad"),
        ("H_OR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/272efcc9-3859-4dd1-8c9d-85ac379c2b9b"),
        ("L_B_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/4dff4e1e-7256-4254-b4cf-eb19907dbcfd"),
        ("L_H_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/83cd2409-6cb5-4b33-ba7c-1efa3b11dfb8"),
        ("L_IR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/b5950444-6730-4b42-a69e-abe919157997"),
        ("L_OR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/2e4947d5-37a3-4b00-bbf6-475b0e5a13b3"),
        ("M1_B_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/99f505be-4e9e-4908-b886-a2129fba81c2"),
        ("M1_H_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/a3213039-9793-44f7-9f1f-633bf1732148"),
        ("M1_IR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/94b9befb-2e42-40d3-974b-895ee6a39999"),
        ("M1_OR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/843f636c-e652-4be6-bf64-ecae5ad425af"),
        ("M2_B_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/4beb0c20-3398-4a25-89d9-bffc606202fc"),
        ("M2_H_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/7ca1ffbc-f1b5-4a38-bd8f-804bf208ee46"),
        ("M2_IR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/feb7c0f4-05af-43d9-97b6-fc3dd6af6702"),
        ("M2_OR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/17ad7f16-6ed9-4bfb-9ac7-360c22e291ee"),
        ("M3_B_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/cc72354a-73b2-4fe9-9be3-0b7f9add4ad6"),
        ("M3_H_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/2348535c-a2b3-4874-8431-ccfe1c088071"),
        ("M3_IR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/88eede73-6694-4000-8454-d0265addf356"),
        ("M3_OR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/91ede68f-32d4-469b-b9f9-c9840b2e41fe"),
        ("U1_B_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/75e22bd3-4558-4338-9606-1987f0c5d725"),
        ("U1_H_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/fc4cb373-139b-4822-b94b-75eb75b4710c"),
        ("U1_IR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/c8f93052-70e2-4d9d-8216-ac0cb4afca79"),
        ("U1_OR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/306854c7-11f5-4c48-a7bc-2a257f3a6d6a"),
        ("U2_B_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/2589631d-6cd9-4e82-bc6e-3d4458462e05"),
        ("U2_H_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/1699d9c5-bcfb-4750-a6b5-ddf06eef01ac"),
        ("U2_IR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/0136afe9-02d3-4279-9390-85ddc9b33f99"),
        ("U2_OR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/5330b494-ea77-4d73-8a55-343779a28e48"),
        ("U3_B_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/54cb586d-6322-41d3-9879-a540ee4e0d7d"),
        ("U3_H_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/234ba0c0-2260-4523-91e0-e74b3f61d98f"),
        ("U3_IR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/031f4dd7-2765-437a-a4ea-9468820ecadd"),
        ("U3_OR_16_6204_1600", "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/3ec17c25-a23d-466a-bbf3-a0665f969279"),
        ("H_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("H_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("H_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("H_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("L_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("L_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("L_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("L_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("M1_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("M1_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("M1_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("M1_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("M2_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("M2_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("M2_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("M2_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("M3_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("M3_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("M3_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("M3_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("U1_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("U1_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("U1_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("U1_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("U2_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("U2_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("U2_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("U2_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("U3_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("U3_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("U3_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("U3_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
("H_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("H_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("H_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("H_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("L_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("L_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("L_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("L_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("M1_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("M1_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("M1_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("M1_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("M2_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("M2_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("M2_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("M2_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("M3_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("M3_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("M3_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("M3_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("U1_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("U1_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("U1_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("U1_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("U2_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("U2_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("U2_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("U2_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("U3_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("U3_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("U3_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("U3_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
("H_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("H_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("H_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("H_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("L_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("L_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("L_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("L_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("M1_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("M1_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("M1_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("M1_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("M2_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("M2_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("M2_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("M2_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("M3_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("M3_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("M3_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("M3_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("U1_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("U1_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("U1_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("U1_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("U2_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("U2_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("U2_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("U2_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("U3_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("U3_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("U3_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("U3_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
("H_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("H_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("H_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("H_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("L_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("L_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("L_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("L_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("M1_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("M1_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("M1_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("M1_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("M2_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("M2_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("M2_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("M2_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("M3_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("M3_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("M3_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("M3_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("U1_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("U1_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("U1_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("U1_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("U2_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("U2_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("U2_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("U2_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        ("U3_B_16_6204_1400", "95083600-04f6-4ac4-854e-d5462f8aefbb"),
        ("U3_H_16_6204_1400", "b4a6118d-ddec-4e66-a2f5-ff58a4e08388"),
        ("U3_IR_16_6204_1400", "4ceb3588-37b6-4f62-baa4-c64cbf0179a5"),
        ("U3_OR_16_6204_1400", "8e8a485f-6fe9-4439-8f93-743a7ac431ec"),
        
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
        return "SU"
    
if __name__ == "__main__":
    dataset = SU()
    print(dataset)
