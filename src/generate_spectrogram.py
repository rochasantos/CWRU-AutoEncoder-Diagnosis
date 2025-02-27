import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def normalize_z_score(signal_data):
    """
    Normalizes the vibration signal using Z-score standardization.
    Centers the signal around mean=0 and standard deviation=1.
    """
    return (signal_data - np.mean(signal_data)) / np.std(signal_data)


def generate_spectrogram(data, output_path, fs=42000, nperseg=512, nfft=1024, noverlap=256):
    
    data = normalize_z_score(data)

    # Compute STFT
    f, t, Sxx = signal.stft(data, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, 
                            nfft=nfft, detrend=False, scaling='spectrum' )

    # Create and save spectrogram
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.imshow(np.log(np.abs(Sxx[: 382, :]**2)), cmap='jet', aspect='auto')
    plt.axis('off')
    plt.gca().invert_yaxis()

    # Save the spectrogram
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100, format='png')
    plt.close(fig)
