import os
import numpy as np
from datasets import CWRU
import librosa
import random

filters = {
        "cwru_ib": {"label": ["I", "B"], "sensor_position": ["no", "centered"], "sampling_rate": "48000", "bearing_type": "6205"},
        "cwru_48k_7_bo": {"label": ["B", "O"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "48000", "bearing_type": "6205", "extent_damage": "007"},
        "cwru_48k_14_bo": {"label": ["B", "O"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "48000", "bearing_type": "6205", "extent_damage": "014"},
        "cwru_48k_21_bo": {"label": ["B", "O"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "48000", "bearing_type": "6205", "extent_damage": "021"},
        "cwru_48k_7_bi": {"label": ["B", "I"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "48000", "bearing_type": "6205", "extent_damage": "007"},
        "cwru_48k_14_bi": {"label": ["B", "I"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "48000", "bearing_type": "6205", "extent_damage": "014"},
        "cwru_48k_21_bi": {"label": ["B", "I"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "48000", "bearing_type": "6205", "extent_damage": "021"},
        "cwru_48k_7_io": {"label": ["I", "O"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "48000", "bearing_type": "6205", "extent_damage": "007"},
        "cwru_48k_14_io": {"label": ["I", "O"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "48000", "bearing_type": "6205", "extent_damage": "014"},
        "cwru_48k_21_io": {"label": ["I", "O"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "48000", "bearing_type": "6205", "extent_damage": "021"}     
    }


def preprocess_signal(x, original_fs, target_fs=48000):
    # Zero Mean
    x = x - np.mean(x)
    std = np.std(x)
    clip_factor = 3.5
    x = np.clip(x, -clip_factor * std, clip_factor * std)
    # Resampling
    if original_fs != target_fs:
        x = librosa.resample(x, orig_sr=original_fs, target_sr=target_fs)
    return x


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


def process_and_save_signal(dataset, filter, output_dir, target_sr=48000, segment_size=9600, max_sample_size=None, augment=False):

    print("[INFO] Starting processing data.")    

    # Get data from the dataset
    for signal, label, original_sr, basename in dataset.load_data(filter):
        max_sample_size = (len(signal) // segment_size) * segment_size if max_sample_size is None else max_sample_size
        # Process the data
        # signal = preprocess_signal(signal, original_sr, target_sr)
        if max_sample_size is not None:
            signal = signal[:max_sample_size]
        # Ensure output directory exists
        os.makedirs(os.path.join(output_dir), exist_ok=True)
        # Save processed signal
        segments = [seg for seg in np.array_split(signal, len(signal) // segment_size)]
        for i, seg in enumerate(segments):
            if augment: # if augmentations
                seg = apply_augmentations(seg)
            output_path = f"{output_dir}/{basename}_{i}"
            # Save signal
            np.save(output_path, {"signal": seg, "label": label})

    print("[INFO] All data has been processed successfully.")



if __name__ == "__main__":
    
    dataset = CWRU()

    # output_dir = os.path.join("data/processed/", filter_key)
    
    folds_idx = [1, 2, 3]
    severities = ["7", "14", "21"]
    for k, severity in enumerate(severities):
        i = k+1
        copy_idx = [*folds_idx]
        test_idx = copy_idx.pop(k)
        train_idx = copy_idx

        filter_key = f"cwru_48k_{severity}_io"
        output_dirs = [
            f"data/processed/{filter_key[-2:]}/fold{test_idx}/test",
            f"data/processed/{filter_key[-2:]}/fold{train_idx[0]}/train",
            f"data/processed/{filter_key[-2:]}/fold{train_idx[1]}/train",
        ]

        for output_dir in output_dirs:
            print(output_dir)
            process_and_save_signal(dataset, filters[filter_key], output_dir, target_sr=48000, segment_size=9600, augment=False)
