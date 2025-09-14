import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import emd



def drop_last_if_residual(imf, fs, threshold_hz=0.5):
    """
    If the last IMF has near-zero median IF, treat it as residual and drop it.
    """
    _, IF, _ = emd.spectra.frequency_transform(imf, fs, 'hilbert')
    med_if = np.nanmedian(IF, axis=1)  # per-IMF median IF over time
    if med_if[-1] < threshold_hz and imf.shape[0] > 1:
        return imf[:-1, :]
    return imf

def ensure_imf_shape(imf, n_samples_expected):
    """
    Ensure IMFs are shaped as (n_imfs, n_samples).
    If we detect (n_samples, n_imfs), transpose them.
    """
    if imf.ndim == 1:
        # Only one trace returned (e.g., residual only) -> make it 2D
        imf = imf[np.newaxis, :]
    if imf.ndim != 2:
        raise ValueError(f"IMF must be 2D, got shape {imf.shape}")
    n0, n1 = imf.shape
    # Heuristic: if the first dim equals n_samples, it's likely (n_samples, n_imfs)
    if n0 == n_samples_expected and n1 != n_samples_expected:
        return imf.T
    return imf

def get_imfs_noise_assisted(x,
                            fs,
                            drop_res=True,
                            max_imfs=5,
                            nensembles=8,
                            ensemble_noise=0.2,
                            nprocesses=None,
                            imf_opts=None):
    """
    Run EEMD (ensemble_sift) with noise assistance and return IMFs shaped (n_imfs, n_samples).
    """
    if imf_opts is None:
        imf_opts = {}

    # --- FIX: coerce None -> 1 so emd doesn't do (None > 1) ---
    if nprocesses is None:
        nprocesses = 1

    imf = emd.sift.ensemble_sift(
        x,
        max_imfs=max_imfs,
        nensembles=nensembles,
        nprocesses=nprocesses,      # now guaranteed int
        ensemble_noise=ensemble_noise,
        imf_opts=imf_opts
    )

    imf = ensure_imf_shape(imf, n_samples_expected=x.size)

    if drop_res:
        imf = drop_last_if_residual(imf, fs)

    return imf


def hht_transform(mixed, fs=42000, freq_range=(0.5, 6_000, 120), sigma=1.0):
    """Plot a smoothed HHT of the mixed signal (skip if near-zero)."""

    N = len(mixed)               # 2048
    t = np.arange(N) / fs

    if mixed.size == 0 or np.allclose(mixed, 0.0) or np.std(mixed) < 1e-12:
        print("[Warn] Mixed is near-zero; skipping HHT.")
        return

    imf_mixed = get_imfs_noise_assisted(
        mixed, fs,
        drop_res=False,
        max_imfs=5,
        nensembles=8,
        ensemble_noise=0.2,
        nprocesses=None,
        imf_opts=None
    )
    _, IFm, IAm = emd.spectra.frequency_transform(imf_mixed, fs, 'hilbert')
    hht_f, hht = emd.spectra.hilberthuang(
        IFm, IAm, freq_range,
        mode='amplitude',
        sum_time=False
    )
    # hht_smooth = gaussian_filter(hht, sigma)
    return hht