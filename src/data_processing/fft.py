import numpy as np
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

