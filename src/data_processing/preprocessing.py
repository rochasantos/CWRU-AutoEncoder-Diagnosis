import numpy as np
from scipy.signal import detrend
from scipy.signal import resample_poly

class Transform:
    """
    Abstract base class for all transformations.
    """
    def apply(self, signal, fs=None):
        raise NotImplementedError("Each transform must implement the apply method.")


class ResampleTransform(Transform):
    def __init__(self, target_fs):     
        self.target_fs = target_fs

    def apply(self, signal, fs):        
        if fs is None:
            raise ValueError("Original sampling rate must be provided for resampling.")

        if fs == self.target_fs:
            # No resampling needed
            return signal

        # Calculate upsample and downsample factors
        gcd = np.gcd(fs, self.target_fs)
        up = self.target_fs // gcd
        down = fs // gcd

        # Apply high-quality resampling
        resampled_signal = resample_poly(signal, up, down)

        return resampled_signal


class NormalizeTransform(Transform):
    def apply(self, signal, fs=None):
        min_val = np.min(signal)
        max_val = np.max(signal)
        normalized_signal = (signal - min_val) / (max_val - min_val)
        return normalized_signal


class OutlierRemovalTransform(Transform):
    def __init__(self, threshold=3.0):
        self.threshold = threshold

    def apply(self, signal, fs=None):
        mean = np.mean(signal)
        std = np.std(signal)
        mask = np.abs(signal - mean) < self.threshold * std
        cleaned_signal = signal[mask]
        return cleaned_signal


class DetrendTransform(Transform):
    def apply(self, signal, fs=None):
        detrended_signal = detrend(signal)
        return detrended_signal


class ZeroMeanTransform(Transform):
    def apply(self, signal, fs=None):
        zero_mean_signal = signal - np.mean(signal)
        return zero_mean_signal


class PreprocessingPipeline:
    """
    A pipeline to apply multiple preprocessing transforms sequentially.
    """
    def __init__(self, transforms=None):
        self.transforms = transforms or []

    def add_transform(self, transform):
        self.transforms.append(transform)

    def apply(self, signal, fs=None):
        for transform in self.transforms:
            signal = transform.apply(signal, fs)
        return signal
