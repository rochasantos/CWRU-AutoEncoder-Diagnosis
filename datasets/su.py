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
        ("M2_B_16_6204_1200", "2122d2dc-349e-4ebd-9edd-be57f079c9be"),
        ("M2_H_16_6204_1200", "c010d5ef-e4cf-475e-b836-387b5a5156cc"),
        ("M2_IR_16_6204_1200", "557cdbab-1930-4917-97fa-d24fce725bbe"),
        ("M2_OR_16_6204_1200", "3c0bfabb-a48b-487e-8aa1-a4a446d5d820"),
        ("M3_B_16_6204_1200", "2f98cb52-5d47-41ff-b0df-00587800b7ec"),
        ("M3_H_16_6204_1200", "c8df65cb-d344-4eb8-9407-e0583798724e"),
        ("M3_IR_16_6204_1200", "9e5cb7ec-ec8c-465d-9351-813452eb969b"),
        ("M3_OR_16_6204_1200", "9215814f-efc3-4de3-90d7-95e234decd16"),
        ("U1_B_16_6204_1200", "a4581e80-13fd-4c1c-9785-f9205ae21636"),
        ("U1_H_16_6204_1200", "bc68ab7c-fc5b-4872-9c47-65680f89defa"),
        ("U1_IR_16_6204_1200", "5493141c-d504-4cf9-9dad-dba81339d86f"),
        ("U1_OR_16_6204_1200", "868693a4-78da-40cf-ab3c-e26e69b0d7bb"),
        ("U2_B_16_6204_1200", "35a1a832-3de2-4346-9760-0217d6f4c589"),
        ("U2_H_16_6204_1200", "bd0585f0-3eaa-4142-8159-6327785aaa80"),
        ("U2_IR_16_6204_1200", "7c3efa04-5c2a-4bcc-a7e1-30bc33b17e7e"),
        ("U2_OR_16_6204_1200", "8be38556-4919-429c-a8f9-04f8dd8de3d4"),
        ("U3_B_16_6204_1200", "a75947c6-462c-497b-87a2-0d23d8cfcc31"),
        ("U3_H_16_6204_1200", "63cac031-b4a7-44c7-8cbb-747c89e1906c"),
        ("U3_IR_16_6204_1200", "d09b1713-0f21-4203-8948-0cea9864eb34"),
        ("U3_OR_16_6204_1200", "c988a856-894b-466b-9285-62e88a96be09"),
        ("H_B_16_6204_1400", "d9eefa63-3d20-4ea4-b3df-32ff37989752"),
        ("H_H_16_6204_1400", "acf09a0c-ed43-4c80-850e-cf65546459be"),
        ("H_IR_16_6204_1400", "db0fcf26-cbd3-4dfd-ad26-c99dd57db84d"),
        ("H_OR_16_6204_1400", "041339a1-0dea-4434-bb79-b78efb62efba"),
        ("L_B_16_6204_1400", "e82d2f48-9a62-45b6-be3d-de1567ac58d3"),
        ("L_H_16_6204_1400", "0ce5b757-13e7-4e5c-b759-8572623c120a"),
        ("L_IR_16_6204_1400", "53537126-c660-4c5b-8080-1d0e12eb1b10"),
        ("L_OR_16_6204_1400", "1c2c26dc-f5ee-4e5f-a86c-9664cb487668"),
        ("M1_B_16_6204_1400", "bf85782f-f276-419a-a0bc-3052b14b2400"),
        ("M1_H_16_6204_1400", "d811a745-f83b-456e-bc11-5185e1b467a6"),
        ("M1_IR_16_6204_1400", "2d51b33c-7ef5-4aae-88c2-e32363ef17fd"),
        ("M1_OR_16_6204_1400", "0a006ed9-123b-4b11-95d0-38c382ffe9e3"),
        ("M2_B_16_6204_1400", "eedea089-4f59-4818-a42f-b19ebed3135c"),
        ("M2_H_16_6204_1400", "d8c71eee-5c9e-46bf-ac04-bf404a47e40f"),
        ("M2_IR_16_6204_1400", "0b2ac6c3-f4e1-459f-840e-ddbf9a491670"),
        ("M2_OR_16_6204_1400", "a308ef87-d2f0-4fc2-be9c-a1a9d85d56a9"),
        ("M3_B_16_6204_1400", "434d1873-6234-4a5c-b77e-27e5cfa14c69"),
        ("M3_H_16_6204_1400", "b94afa4c-1e5d-4030-bc34-776ca10581b8"),
        ("M3_IR_16_6204_1400", "581b5fb5-1bf9-41d5-b1d3-8b5e561488ff"),
        ("M3_OR_16_6204_1400", "8b025537-e3f5-4063-b3c6-df107e661167"),
        ("U1_B_16_6204_1400", "2cb335f8-144c-42b9-8fbc-5707a7719509"),
        ("U1_H_16_6204_1400", "a4dbb3db-9523-468b-a99e-229f38b89d52"),
        ("U1_IR_16_6204_1400", "790ccf3c-a2ca-4b88-b5ae-95dd37a316fb"),
        ("U1_OR_16_6204_1400", "0adb3ce2-e40e-45c7-b5c4-dc30aa5c8dd0"),
        ("U2_B_16_6204_1400", "fa7f4159-8291-447d-a7c0-9b998ed513ea"),
        ("U2_H_16_6204_1400", "3d39dd9b-4e5f-4f6c-8e7b-67e378434799"),
        ("U2_IR_16_6204_1400", "0f381a3a-d2cc-4e6b-ba60-ad36c98c7395"),
        ("U2_OR_16_6204_1400", "1386ae80-363e-43d7-bfed-39bcbddde696"),
        ("U3_B_16_6204_1400", "4ef27cc1-7e7d-4cfa-8faf-bbc885eac179"),
        ("U3_H_16_6204_1400", "b83dea15-9dbd-41a2-aacc-b7d195a27027"),
        ("U3_IR_16_6204_1400", "68383fa8-da64-45fc-bc90-6363489b1cdf"),
        ("U3_OR_16_6204_1400", "9c35a5f2-3877-41e9-8195-950e5e803bff"),
        ("H_B_16_6204_1600", "f4c148d3-3eb9-44f4-bbb0-fafd03bf106b"),
        ("H_H_16_6204_1600", "c0a24d70-6f3c-46ac-b1f8-94214229c20d"),
        ("H_IR_16_6204_1600", "6d67ab70-b08a-4961-b3ae-5a54ff3b80ad"),
        ("H_OR_16_6204_1600", "272efcc9-3859-4dd1-8c9d-85ac379c2b9b"),
        ("L_B_16_6204_1600", "4dff4e1e-7256-4254-b4cf-eb19907dbcfd"),
        ("L_H_16_6204_1600", "83cd2409-6cb5-4b33-ba7c-1efa3b11dfb8"),
        ("L_IR_16_6204_1600", "b5950444-6730-4b42-a69e-abe919157997"),
        ("L_OR_16_6204_1600", "2e4947d5-37a3-4b00-bbf6-475b0e5a13b3"),
        ("M1_B_16_6204_1600", "99f505be-4e9e-4908-b886-a2129fba81c2"),
        ("M1_H_16_6204_1600", "a3213039-9793-44f7-9f1f-633bf1732148"),
        ("M1_IR_16_6204_1600", "94b9befb-2e42-40d3-974b-895ee6a39999"),
        ("M1_OR_16_6204_1600", "843f636c-e652-4be6-bf64-ecae5ad425af"),
        ("M2_B_16_6204_1600", "4beb0c20-3398-4a25-89d9-bffc606202fc"),
        ("M2_H_16_6204_1600", "7ca1ffbc-f1b5-4a38-bd8f-804bf208ee46"),
        ("M2_IR_16_6204_1600", "feb7c0f4-05af-43d9-97b6-fc3dd6af6702"),
        ("M2_OR_16_6204_1600", "17ad7f16-6ed9-4bfb-9ac7-360c22e291ee"),
        ("M3_B_16_6204_1600", "cc72354a-73b2-4fe9-9be3-0b7f9add4ad6"),
        ("M3_H_16_6204_1600", "2348535c-a2b3-4874-8431-ccfe1c088071"),
        ("M3_IR_16_6204_1600", "88eede73-6694-4000-8454-d0265addf356"),
        ("M3_OR_16_6204_1600", "91ede68f-32d4-469b-b9f9-c9840b2e41fe"),
        ("U1_B_16_6204_1600", "75e22bd3-4558-4338-9606-1987f0c5d725"),
        ("U1_H_16_6204_1600", "fc4cb373-139b-4822-b94b-75eb75b4710c"),
        ("U1_IR_16_6204_1600", "c8f93052-70e2-4d9d-8216-ac0cb4afca79"),
        ("U1_OR_16_6204_1600", "306854c7-11f5-4c48-a7bc-2a257f3a6d6a"),
        ("U2_B_16_6204_1600", "2589631d-6cd9-4e82-bc6e-3d4458462e05"),
        ("U2_H_16_6204_1600", "1699d9c5-bcfb-4750-a6b5-ddf06eef01ac"),
        ("U2_IR_16_6204_1600", "0136afe9-02d3-4279-9390-85ddc9b33f99"),
        ("U2_OR_16_6204_1600", "5330b494-ea77-4d73-8a55-343779a28e48"),
        ("U3_B_16_6204_1600", "54cb586d-6322-41d3-9879-a540ee4e0d7d"),
        ("U3_H_16_6204_1600", "234ba0c0-2260-4523-91e0-e74b3f61d98f"),
        ("U3_IR_16_6204_1600", "031f4dd7-2765-437a-a4ea-9468820ecadd"),
        ("U3_OR_16_6204_1600", "3ec17c25-a23d-466a-bbf3-a0665f969279"),
        ("H_B_16_6204_600", "a996c873-401a-43ca-baa3-fb412922cd9e"),
        ("H_H_16_6204_600", "6c26fbac-7b11-4c6e-818c-650cbd0069f7"),
        ("H_IR_16_6204_600", "dbc1432c-217e-4dca-82e3-45d5bc43c7d7"),
        ("H_OR_16_6204_600", "047efc61-62e5-4b09-a102-96833c43e49f"),
        ("L_B_16_6204_600", "44b88535-ccb9-4c56-a82a-2f268e1e04dd"),
        ("L_H_16_6204_600", "d8c4cac0-bf31-4c21-a259-d680a16e3d45"),
        ("L_IR_16_6204_600", "51c15953-f812-4ba3-8444-b584427692af"),
        ("L_OR_16_6204_600", "b207b518-23ea-4ec8-aa74-f178a77e3cd7"),
        ("M1_B_16_6204_600", "a025a5ef-7f41-4ab2-9042-f6979a344d81"),
        ("M1_H_16_6204_600", "68219e40-6d8c-49d5-b6ff-3332c802854b"),
        ("M1_IR_16_6204_600", "731ba6a9-f5c3-43fd-87dd-76f0681038da"),
        ("M1_OR_16_6204_600", "7bab429b-e14f-4ee3-9bc4-be2ccdc11d57"),
        ("M2_B_16_6204_600", "37b7c1dc-c726-4154-9418-8f6ed15489fe"),
        ("M2_H_16_6204_600", "ef1b6242-aabf-4159-a9bf-bbc690e946f9"),
        ("M2_IR_16_6204_600", "e0600217-3c6f-4fce-8c70-a98f431108bf"),
        ("M2_OR_16_6204_600", "b0f5b5b3-ff53-4a95-9729-2d4b56f9d758"),
        ("M3_B_16_6204_600", "f586e093-7263-4f42-b0a3-6e1ec25a2c09"),
        ("M3_H_16_6204_600", "72989081-6170-4f28-9d43-bcbf6bb74e76"),
        ("M3_IR_16_6204_600", "9067eced-415c-4181-b8e7-bcf96e2472bb"),
        ("M3_OR_16_6204_600", "48e98e98-006c-4fc5-bfb2-27735f62ead7"),
        ("U1_B_16_6204_600", "72a01a77-ad5f-4984-b523-d992fe4129c5"),
        ("U1_H_16_6204_600", "11081283-327d-4036-9bc0-0958c59a7e4e"),
        ("U1_IR_16_6204_600", "43cf6056-9513-4d1f-86cb-c331faf8ceb4"),
        ("U1_OR_16_6204_600", "70a15963-00c1-4893-928b-c5e01c0fc750"),
        ("U2_B_16_6204_600", "b52efe53-c4a0-46b8-9eb7-e9c723b5132b"),
        ("U2_H_16_6204_600", "b5355ef0-6401-41c6-a6f0-09b7f2bcaec4"),
        ("U2_IR_16_6204_600", "c16f467b-094e-47f6-8915-a9bc51ae6357"),
        ("U2_OR_16_6204_600", "d315ce09-d8c0-413a-adec-dab804bb0ba4"),
        ("U3_B_16_6204_600", "55e74212-a7b1-4d31-a4e5-45cafd7359db"),
        ("U3_H_16_6204_600", "c57d93c3-c0d6-4247-96d4-a652b5837d43"),
        ("U3_IR_16_6204_600", "51414c2b-0913-4871-a2a5-0801ff95d6b4"),
        ("U3_OR_16_6204_600", "dd59306a-7651-44ce-949e-7e4d42f12cdd"),
        ("H_B_16_6204_800", "92f79b4c-f966-4e1e-a12e-7b035d208a32"),
        ("H_H_16_6204_800", "2ce19fbf-04cb-479b-ac27-03d80fdb2559"),
        ("H_IR_16_6204_800", "8e7a97e8-75e0-4692-b48f-fc36bd82942c"),
        ("H_OR_16_6204_800", "5baef7b9-df16-4de8-946a-467335f70373"),
        ("L_B_16_6204_800", "084ffb6d-50eb-4dfd-9cb2-e734f3786f22"),
        ("L_H_16_6204_800", "2c645dd1-4f2b-4fac-a359-abfbb2a7a53a"),
        ("L_IR_16_6204_800", "19bfc77f-7a9d-4da8-99f1-9f78974be649"),
        ("L_OR_16_6204_800", "fce85fbf-9d85-405c-b7ce-c0a13c4456f1"),
        ("M1_B_16_6204_800", "5e89bd6c-e73f-401f-9e22-be126fcba494"),
        ("M1_H_16_6204_800", "1fec1969-5714-4a26-9725-c2686611504a"),
        ("M1_IR_16_6204_800", "3d49c6e9-e909-4b4c-8ec3-e108afc7ad15"),
        ("M1_OR_16_6204_800", "dfece10d-05d0-4472-8c33-f2aeaf7a9e56"),
        ("M2_B_16_6204_800", "c770b984-ac2f-4a14-88e6-8fe040d0ce51"),
        ("M2_H_16_6204_800", "477088f2-e717-4377-a97f-b766421d73db"),
        ("M2_IR_16_6204_800", "7c5b7876-6893-493f-a492-c08f19240db1"),
        ("M2_OR_16_6204_800", "89c25cf7-f5aa-4860-ae11-06b28da052b1"),
        ("M3_B_16_6204_800", "ff1fa54d-4fd4-4133-8d2c-c65d64e167c0"),
        ("M3_H_16_6204_800", "ffd40879-d22c-4f74-8bb1-4bac950db04c"),
        ("M3_IR_16_6204_800", "11459543-204e-472d-bacf-47a3b472d578"),
        ("M3_OR_16_6204_800", "b47bcb86-0fd9-41a7-b902-29c7422068f9"),
        ("U1_B_16_6204_800", "e93f8860-d64f-4441-be0c-720d88fc409b"),
        ("U1_H_16_6204_800", "19f971aa-1474-4e80-9b9e-fc6086f22252"),
        ("U1_IR_16_6204_800", "aa94bdd1-92b7-435c-ac99-6f34aa3d970d"),
        ("U1_OR_16_6204_800", "07645077-677d-42c1-9fd2-d34628752431"),
        ("U2_B_16_6204_800", "e825b68c-c6e8-4d98-9524-63f306fd0940"),
        ("U2_H_16_6204_800", "35f8ab45-20dc-4444-a701-77d59bd14392"),
        ("U2_IR_16_6204_800", "05c97b2f-3f48-47c3-8c8e-2c19e83121af"),
        ("U2_OR_16_6204_800", "c8471aa8-6e9e-4dcd-807b-94161dfc87d5"),
        ("U3_B_16_6204_800", "df3a8729-a034-4aa1-bbb3-f4d567dbb7dd"),
        ("U3_H_16_6204_800", "68922486-394c-4190-9f06-4fb04f10965c"),
        ("U3_IR_16_6204_800", "d3fc4405-851a-4b94-b96d-cfdb6b456701"),
        ("U3_OR_16_6204_800", "32d12957-7e36-4895-a684-5de9573abca5"),
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
        data = matlab_file["Data"][:, 0]
         
        if self.acquisition_maxsize:
            return data[:self.acquisition_maxsize], label
        else:
            return data, label

    def __str__(self):
        return "SU"
    
if __name__ == "__main__":
    dataset = SU()
    print(dataset)
