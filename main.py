import sys
import logging
from src.utils import LoggerWriter
import os
import numpy as np
from scipy import signal
from datasets import CWRU, UORED, Hust, Paderborn
from experimenter_spectrogram.main import experimenter
from experimenter_spectrogram.generate_spectrogram import generate_spectrogram

def create_directory_structure():
    root_dir = "data/spectrograms"
    for severity in ["007", "014", "021"]:
        for label in ["N", "I", "O", "B"]:
            dir_path = os.path.join(root_dir, severity, label)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

def download():
    Paderborn().download()

def create_spectrogram():
    print("Creating the spectrograms.")
    for label in ["I", "O", "B"]:
        directories = [
            # f"data/processed/cwru/007/{label}",
            # f"data/processed/cwru/014/{label}",
            # f"data/processed/cwru/021/{label}",
            f"data/processed/hust/{label}",
            # f"data/processed/uored/{label}",
            # f"data/processed/cwru_hust/{label}",
            # f"data/processed/cwru_uored/{label}",
        ]    

        # creates a directory structure for the augmented data.
        root_dir = f"data/spectrogram/hust/{label}"
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        for dir in directories:
            # load_severity = dir[-5:-2]
            for j, file in enumerate(os.listdir(dir)):
                data = np.load(os.path.join(dir, file))
                signal, _ = data[:-1], data[-1]
                output_path = os.path.join(root_dir,f"spectro_{j}.png")
                if os.path.exists(output_path):
                    continue
                generate_spectrogram(signal, output_path)
            
    print("finish!")           


if __name__ == "__main__":
    sys.stdout = LoggerWriter(logging.info, "cwru_uored_hust")


    # Create structure of directories
    # create_directory_structure()

    # download
    # download()
    experimenter()
    # create_spectrogram()
    