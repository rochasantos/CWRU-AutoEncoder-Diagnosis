import os
import numpy as np
from src.data_processing.spectrogram import generate_spectrogram


def create_spectrogram():
    print("Creating the spectrograms.")
    directories = [
        f"data/processed/cwru/007",
        f"data/processed/cwru/014",
        f"data/processed/cwru/021",
        # f"data/processed/hust",
        # f"data/processed/uored",
        # f"data/processed/paderborn",
    ]
    for dir in directories:
        classes_name = ["I", "O"] if dir.split('/')[-1] == 'paderborn' else ["N", "I", "O", "B"]
        for label in classes_name:
            output_dir = dir.replace("processed", "spectrogram")+'/'+label
            print(f"Creating spectrograms in {output_dir}")
            root_dir = os.path.join(dir, label)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for i, file in enumerate(os.listdir(root_dir)):
                data = np.load(os.path.join(root_dir, file))
                signal, _ = data[:-1], data[-1]
                output_path = os.path.join(output_dir,f"spectro_{i}.png")
                if os.path.exists(output_path):
                    continue
                generate_spectrogram(signal, output_path)
    print("All spectrograms created successfully.")