import os
from scipy import signal
from datasets import CWRU
from scripts import generate_spectrogram

def create_directory_structure():
    root_dir = "data/spectrograms"
    for severity in ["007", "014", "021"]:
        for label in ["N", "I", "O", "B"]:
            dir_path = os.path.join(root_dir, severity, label)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")


def create_spectrograms():    
    dataset = CWRU()
    segment_length = 48000
    filter_parameters = {
        "label": ["N", "I", "O", "B"],
        "sampling_rate": "48000",
    }  
    hp_severity_map = {"0": "007", "1": "014", "2": "021"}
    for extent_damage in ["000", "007", "014", "021"]:
        metainfo = dataset.metainfo.filter_data({**filter_parameters, "extent_damage": extent_damage})        
        for info in metainfo:
            severity = extent_damage
            if info["label"] == "N":
                if info["hp"] not in hp_severity_map:
                    continue
                severity = hp_severity_map[info["hp"]]
            basename = info["filename"]
            filepath = os.path.join('data/raw/', dataset.__class__.__name__.lower(), basename+'.mat')
            data, label = dataset.load_signal_by_path(filepath)
            detrended_data = signal.detrend(data)
            for i in range(data.shape[0] // segment_length):
                sample = detrended_data[i*segment_length:(i+1)*segment_length]
                output_path = f"data/temp/{severity}/{label}/{basename}_{i}.png"   
                # Creates and saves the spectrogram in a directory according to its classification.
                if not os.path.exists(output_path):
                    generate_spectrogram(data=sample, 
                                        output_path=output_path,
                                        nfft=1024, fs=48000, nperseg=697, noverlap=442)            



if __name__ == "__main__":
    # Create structure of directories
    # create_directory_structure()

    # create spectrograms
    create_spectrograms()
    