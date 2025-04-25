import os
import numpy as np
from matplotlib import colormaps
from PIL import Image
from scipy.signal import stft


def compute_spectrogram(signal, fs, nperseg=256, noverlap=128, nfft=1600):
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    Sxx = np.abs(Zxx)
    logSxx = np.log(Sxx**2 + 1e-10)

def save_images(signal, output_path, cmap_name='jet'):
    colormap = colormaps[cmap_name]
    rgba_img = colormap(signal)
    rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)  # Discard alpha
    img = Image.fromarray(rgb_img)
    img.save(output_path)

def create_spectrogram(datasets_dir, fs, nperseg=256, noverlap=128, nfft=1600):
    print("Starting spectrogram creation.")
    for dir in datasets_dir:        
        classes = os.listdir(dir)
        for label in classes:
            root_dir = os.path.join(dir, label)
            output_dir = root_dir.replace("processed", "spectrogram")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for file in os.listdir(root_dir):
                if file == "N15_M07_F04_K006_8.npy":  # optional exclusion
                    continue
                filepath = os.path.join(root_dir, file)
                signal = np.load(filepath)
                output_path = filepath.replace(root_dir, output_dir).replace(".npy", ".png")
                save_images(signal, output_path, cmap_name='jet')

    print("All spectrograms created successfully.")


if __name__ == "__main__":
    datasets_dir = [
        "data/processed/cwru/007",
        # "data/processed/cwru/014",
        # "data/processed/cwru/021",
        # "data/processed/hust",
        # "data/processed/uored",
        # "data/processed/paderborn",
    ]
    create_spectrogram(datasets_dir=datasets_dir)
