import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def generate_spectrogram(data, output_path, fs, nperseg=256, nfft=None, noverlap=None):
    
    # Compute STFT
    f, t, Sxx = signal.stft(data, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, 
                            nfft=nfft, detrend=False )

    # Create and save spectrogram
    fig = plt.figure(figsize=(10/2, 6/2))
    plt.imshow(np.log(np.abs(Sxx[: 382, :]**2)), cmap='jet', aspect='auto')
    plt.axis('off')
    plt.gca().invert_yaxis()

    # Save the spectrogram
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)




def generate_spectrogram2(data, output_path, fs, nperseg=256, nfft=None, noverlap=None, output_size=(128, 128)):
    """
    Generate and save a high-resolution spectrogram.

    Parameters:
    - data: Input signal data.
    - output_path: Path to save the generated spectrogram.
    - fs: Sampling frequency of the input signal.
    - nperseg: Number of samples per segment.
    - nfft: Number of FFT points.
    - noverlap: Number of overlapping points between segments.
    - output_size: Tuple specifying the desired output image size (width, height).
    """
    # Compute STFT
    f, t, Sxx = signal.stft(data, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, 
                            nfft=nfft, detrend=False)
    
    # Create figure and plot the spectrogram
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(np.log(np.abs(Sxx[:382, :]**2)), cmap='jet', aspect='auto')
    plt.axis('off')
    plt.gca().invert_yaxis()
    
    # Save the spectrogram with high resolution
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=128)
    plt.close(fig)