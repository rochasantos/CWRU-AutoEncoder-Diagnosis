import os
import numpy as np
from src.data_processing.spectrogram import generate_spectrogram


def create_spectrogram():
    print("Creating the spectrograms.")
    directories = [
        f"data/processed/cwru/007",
        f"data/processed/cwru/014",
        f"data/processed/cwru/021",
        f"data/processed/hust",
        f"data/processed/uored",
    ]
    for dir in directories:
        for label in ["I", "O", "B"]:
            output_dir = dir.replace("processed", "spectrogram")
            print(f"Creating spectrograms in {output_dir}")
            root_dir = os.path.join(output_dir, label)
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
            for i, file in enumerate(os.listdir(dir)):
                data = np.load(os.path.join(dir, file))
                signal, _ = data[:-1], data[-1]
                output_path = os.path.join(root_dir,f"spectro_{i}.png")
                if os.path.exists(output_path):
                    continue
                generate_spectrogram(signal, output_path)
    print("All spectrograms created successfully.")