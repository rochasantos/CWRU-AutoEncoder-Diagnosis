import os
import sys
import logging
from src.utils import LoggerWriter

from scripts.download import download
from scripts.experimenter import experimenter
from scripts.create_spectrogram import create_spectrogram


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
    # download()
    # create_spectrogram()    
    experimenter()
    