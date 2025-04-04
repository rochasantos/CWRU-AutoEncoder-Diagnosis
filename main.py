import os
import sys
import logging
from src.utils import LoggerWriter

from scripts.download import download
from scripts.process_signal import process_signal
from scripts.create_spectrogram import create_spectrogram
from scripts.experimenter_spectronet import experimenter
from scripts.spectrogram_chunker import generate_chunk

def create_directory_structure():
    root_dir = "data/spectrograms"
    for severity in ["007", "014", "021"]:
        for label in ["N", "I", "O", "B"]:
            dir_path = os.path.join(root_dir, severity, label)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")


if __name__ == "__main__":
    sys.stdout = LoggerWriter(logging.info, "log")

    # create_directory_structure()
    # download("su")
    # process_signal(dataset_name="cwru", target_sr=42000)
    # create_spectrogram()   
    # generate_chunk()
    experimenter()
    