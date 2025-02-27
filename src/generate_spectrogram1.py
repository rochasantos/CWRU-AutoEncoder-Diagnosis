import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import cv2

def generate_spectrogram(signal_data, fs, output_path, target_size=(224, 224), colormap=cv2.COLORMAP_INFERNO):
    """
    Generates a spectrogram from a time-series signal and saves it as a 224x224 color image (RGB),
    maintaining aspect ratio with padding.
    
    Parameters:
    - signal_data: numpy array containing the time-series signal.
    - fs: Sampling frequency of the signal.
    - output_path: Path to save the spectrogram image.
    - target_size: Tuple (width, height) for final output.
    - colormap: OpenCV colormap for color conversion (default: INFERNO).
    """
    # Compute STFT
    f, t, Zxx = signal.stft(signal_data, fs=fs, nperseg=256, noverlap=128)
    
    # Convert to dB scale
    spectrogram_db = 20 * np.log10(np.abs(Zxx) + 1e-8)  # Avoid log(0)

    # Normalize between 0 and 255 for visualization
    spectrogram_db = 255 * (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())
    spectrogram_db = spectrogram_db.astype(np.uint8)  # Convert to uint8

    # Resize while keeping aspect ratio
    original_size = spectrogram_db.shape[::-1]  # (width, height)
    scale_factor = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
    new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))

    resized_img = cv2.resize(spectrogram_db, new_size, interpolation=cv2.INTER_AREA)

    # Apply padding to reach the target size
    delta_w = target_size[0] - new_size[0]
    delta_h = target_size[1] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # Apply colormap to convert grayscale to RGB
    colored_img = cv2.applyColorMap(padded_img, colormap)

    # Save as PNG
    cv2.imwrite(output_path, colored_img)
