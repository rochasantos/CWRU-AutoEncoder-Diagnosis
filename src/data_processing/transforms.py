import numpy as np
from scipy.signal import stft
from scipy.signal import detrend, windows

def fft_signal(signal, fs=48000):
    signal = signal[:,0]
    # Pre-processing
    signal = detrend(signal)
    window = windows.hann(len(signal))
    signal = signal * window

    # FFT
    N = len(signal)
    freq = np.fft.fftfreq(N, d=1/fs)
    fft_values = np.fft.fft(signal)

    # Positive frequencies and normalization
    mask = freq >= 0
    freq = freq[mask]
    fft_magnitude = np.abs(fft_values[mask]) * 2 / N
    return fft_magnitude


def stft_transform(signal_1d, nperseg=64, noverlap=32, nfft=128):
    """
    Aplica a STFT em um sinal 1D e retorna o espectrograma 2D (magnitude).
    """
    assert len(signal_1d) == 1024, "O sinal deve ter exatamente 1024 pontos."

    # Normalização
    signal_1d = (signal_1d - np.mean(signal_1d)) / np.std(signal_1d)

    # STFT
    f, t, Zxx = stft(signal_1d, fs=1.0, nperseg=nperseg, noverlap=noverlap, nfft=nfft, padded=False, boundary=None)

    # Magnitude do espectrograma
    spectrogram = np.abs(Zxx)

    return spectrogram


def transform_2d(signal_1d):
    # signal_1d = (signal_1d - np.mean(signal_1d)) / np.std(signal_1d)    
    image_2d = signal_1d.reshape(1, 32, 32)
    return image_2d

from scipy.fft import rfft

def compute_rms_fft(signal, fs=48000, n_fft=4096):
    window = np.hanning(n_fft)
    signal = signal[:n_fft] * window
    fft_vals = rfft(signal)
    fft_magnitude = np.abs(fft_vals) * 2 / np.sum(window)
    rms_spectrum = fft_magnitude / np.sqrt(2)
    return rms_spectrum.astype(np.float32)
