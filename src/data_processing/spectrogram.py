import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, detrend

def generate_spectrogram(signal, output_path, fs=42000, nperseg=200, noverlap=192, nfft=1600):
    # Remove trend from the signal
    signal = detrend(signal)

    # Compute STFT
    _, _, Sxx = stft(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft
    )

    # Convert to log scale
    log_Sxx = np.log(np.abs(Sxx) ** 2 + 1e-10)

    # Normalize to 0-255
    normalized = 255 * (log_Sxx - log_Sxx.min()) / (log_Sxx.max() - log_Sxx.min())
    gray_img = normalized.astype(np.uint8)

    # Apply colormap and convert to RGB
    bgr_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # Save image
    plt.imsave(output_path, rgb_img)
