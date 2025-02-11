import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def generate_spectrogram(data, output_path, fs, nperseg=256, nfft=None, noverlap=None):
    
    # Compute STFT
    f, t, Sxx = signal.stft(data, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, 
                            nfft=nfft, detrend=False )

    # Create and save spectrogram
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(np.log(np.abs(Sxx[: 382, :]**2)), cmap='jet', aspect='auto')
    plt.axis('off')
    plt.gca().invert_yaxis()

    # Save the spectrogram
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
