import os
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
from datasets import CWRU, Paderborn
import src.data_processing.preprocessing as preprocessing

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa
import random

filters = {
        "cwru": {"label": ["N", "I", "O", "B"], "sensor_position": ["no", "centered"], "sampling_rate": "48000", "bearing_type": "6205"},
        "cwru_48k_7": {"label": ["N", "I", "O", "B"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "48000", "bearing_type": "6205", "extent_damage": "007"},
        "cwru_48k_14": {"label": ["N", "I", "O", "B"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "48000", "bearing_type": "6205", "extent_damage": "014"},
        "cwru_48k_21": {"label": ["N", "I", "O", "B"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "12000", "bearing_type": "6205", "extent_damage": "021"},
        "cwru_12k_7": {"label": ["N", "I", "O", "B"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "12000", "bearing_type": "6205", "extent_damage": "007"},
        "cwru_12k_14": {"label": ["N", "I", "O", "B"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "12000", "bearing_type": "6205", "extent_damage": "014"},
        "cwru_12k_21": {"label": ["N", "I", "O", "B"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "12000", "bearing_type": "6205", "extent_damage": "021"},
        "cwru_ref": {"label": "N", "bearing_type": "6205"},
        "uored": {"label": ["N", "I", "O", "B"], "condition_bearing_health": ["faulty", "healthy"]},
        "hust": {"label": ["N", "I", "O", "B"], "bearing_type": ["6204", "6205", "6206", "6207", "6208"]},
        "hust_4": {"label": ["N", "I", "O", "B"], "bearing_type": "6204"},
        "hust_5": {"label": ["N", "I", "O", "B"], "bearing_type": "6205"},
        "hust_6": {"label": ["N", "I", "O", "B"], "bearing_type": "6206"},
        "hust_7": {"label": ["N", "I", "O", "B"], "bearing_type": "6207"},
        "hust_8": {"label": ["N", "I", "O", "B"], "bearing_type": "6208"},
        "paderborn_artificial": [
            "K004", "K005", "K006",
            "KA01", "KA03", "KA05", "KA06", "KA07", "KA09", 
            "KI01", "KI03", "KI07", "KI08"
        ],
        "paderborn_real": [
            "K001", "K002", "K003", 
            "KA04", "KA15", "KA16", "KA22", "KA30" 
            "KI04", "KI14", "KI16", "KI18", "KI21"
        ],
        "su": {"label": ["N", "I", "O", "B"], "condition_bearing_health": ["faulty", "healthy"]},
    }


def preprocess_signal(x, original_fs, target_fs=48000):
    # Zero Mean
    x = x - np.mean(x)
    # Zero Mean 
    std = np.std(x)
    clip_factor = 3.5
    x = np.clip(x, -clip_factor * std, clip_factor * std)
    # Resampling
    if original_fs != target_fs:
        x = librosa.resample(x, orig_sr=original_fs, target_sr=target_fs)
    return x


def compute_stft_segment(signal_data, fs=48000, nperseg=256, noverlap=128, nfft=256):
    f, t, Zxx = signal.stft(signal_data, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    Sxx = np.abs(Zxx)
    if Sxx.shape[0] != 129:
        raise ValueError(f"Expected frequency bins=129, got {Sxx.shape[0]}")
    return Sxx


def split_into_segments(spectrogram, segment_width=129):
    segments = []
    _, total_width = spectrogram.shape
    for start in range(0, total_width - segment_width + 1, segment_width):
        segment = spectrogram[:, start:start + segment_width]
        segments.append(segment)
        if len(segment) == 12:
            return segments
    return segments


def normalize_segment(segment):
    min_val = np.min(segment)
    max_val = np.max(segment)
    if max_val - min_val == 0:
        return np.zeros_like(segment)
    return (segment - min_val) / (max_val - min_val)


def apply_augmentations(segment):
    if random.random() < 0.5:
        segment = np.fliplr(segment) # horizontal flip
    if random.random() < 0.5:
        segment = np.flipud(segment) # vertical flip
    if random.random() < 0.5:
        segment += np.random.normal(0, 0.001, segment.shape) # add noise
    if random.random() < 0.5:
        shift = random.randint(-5, 5) # random shift
        segment = np.roll(segment, shift, axis=0)
    return segment


def save_segment_as_image(segment, save_path, cmap='inferno'):
    plt.imsave(save_path, segment, cmap=cmap, format='png', origin='lower')


def process_and_save_spectrogram(dataset, filter, output_dir, target_sr=48000, augment=True):

    print("[INFO] Starting spectrogram creation.")    

    # Get data from the dataset
    for signal, label, original_sr, basename in dataset.load_data(filter):
        # Process the data
        signal = preprocess_signal(signal, original_sr, target_sr)
        # Ensure output directory exists
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)
        # Save spectrogram
        spectrogram = compute_stft_segment(signal)
        segments = split_into_segments(spectrogram)
        for i, seg in enumerate(segments):
            if augment:
                seg = apply_augmentations(seg)
            seg = normalize_segment(seg)
            output_path = f"{output_dir}/{label}/{basename}_{i}.png"
            save_segment_as_image(seg, output_path)

    print("[INFO] All spectrograms created successfully.")



if __name__ == "__main__":
    
    dataset = Paderborn()
    filter_key = "paderborn_artificial"

    output_dir = os.path.join("data/spectrograms", filter_key)
    process_and_save_spectrogram(dataset, filters[filter_key], output_dir, target_sr=48000, augment=True)
