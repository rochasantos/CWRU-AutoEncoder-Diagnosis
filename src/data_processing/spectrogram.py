import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, detrend


def generate_spectrogram(signal, output_path, fs=42000, nperseg=200, noverlap=192, nfft=1600):
    spectrogram_generator = SpectrogramGenerator(window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    output = spectrogram_generator.generate(signal, sample_rate=fs)

    plt.imshow(output["spectrogram"])
    plt.axis("off")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100, format='png')
    plt.imsave(output_path, output, cmap="jet")
    plt.close()


class SpectrogramGenerator:

    def __init__(self, window="hann", nperseg=None, noverlap=None, nfft=None):
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft

    def generate(self, signal, sample_rate):
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

        return rgb_img
