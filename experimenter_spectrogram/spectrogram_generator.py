import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import stft, detrend

class SpectrogramGenerator:
    """Generates a spectrogram from a raw vibration signal.

    Parameters:
        window (str): Type of window function to apply (e.g., 'hann').
        nperseg (int): Number of samples per segment.
        noverlap (int): Number of overlapping samples.
        nfft (int): Number of points in FFT.
    """

    def __init__(self, window="hann", nperseg=None, noverlap=None, nfft=None):
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft

    def generate(self, signal, sample_rate, label):
        """Computes the spectrogram of the input signal and returns an image.

        Args:
            signal (np.array): 1D array representing the vibration signal.
            sample_rate (int): Sampling rate of the signal.
            label (str): Label associated with the signal.

        Returns:
            dict: Containing the spectrogram image and its corresponding label.
        """
        # Remove signal trends
        signal = detrend(signal)

        # Compute STFT
        f, t, Sxx = stft(
            signal,
            fs=sample_rate,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft
        )

        # Filter out frequencies above 10 kHz
        max_freq = 10000  # 10 kHz
        max_bound = np.argmin(np.abs(f - max_freq))
        Sxx = Sxx[: max_bound + 1, :]

        # Convert to log scale
        log_Sxx = np.log(np.abs(Sxx) ** 2 + 1e-10)  # Adding a small value to avoid log(0)

        # Normalize the spectrogram to 0-255
        normalized = ((log_Sxx - log_Sxx.min()) / (log_Sxx.max() - log_Sxx.min())) * 255
        gray_img = normalized.astype(np.uint8)

        # Convert grayscale image to RGB using colormap
        bgr_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        return {"spectrogram": rgb_img, "label": label}
