import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import emd


# # -----------------------
# # Helpers
# # -----------------------
# def ensure_imf_shape(imf, n_samples_expected):
#     """
#     Ensure IMFs are shaped as (n_imfs, n_samples).
#     If we detect (n_samples, n_imfs), transpose them.
#     """
#     if imf.ndim == 1:
#         # Only one trace returned (e.g., residual only) -> make it 2D
#         imf = imf[np.newaxis, :]
#     if imf.ndim != 2:
#         raise ValueError(f"IMF must be 2D, got shape {imf.shape}")
#     n0, n1 = imf.shape
#     # Heuristic: if the first dim equals n_samples, it's likely (n_samples, n_imfs)
#     if n0 == n_samples_expected and n1 != n_samples_expected:
#         return imf.T
#     return imf


# def drop_last_if_residual(imf, fs, threshold_hz=0.5):
#     """
#     If the last IMF has near-zero median IF, treat it as residual and drop it.
#     """
#     _, IF, _ = emd.spectra.frequency_transform(imf, fs, 'hilbert')
#     med_if = np.nanmedian(IF, axis=1)  # per-IMF median IF over time
#     if med_if[-1] < threshold_hz and imf.shape[0] > 1:
#         return imf[:-1, :]
#     return imf


# def get_imfs_noise_assisted(x,
#                             fs,
#                             drop_res=True,
#                             max_imfs=5,
#                             nensembles=8,
#                             ensemble_noise=0.2,
#                             nprocesses=None,
#                             imf_opts=None):
#     """
#     Run EEMD (ensemble_sift) with noise assistance and return IMFs shaped (n_imfs, n_samples).
#     """
#     if imf_opts is None:
#         imf_opts = {}

#     # --- FIX: coerce None -> 1 so emd doesn't do (None > 1) ---
#     if nprocesses is None:
#         nprocesses = 1

#     imf = emd.sift.ensemble_sift(
#         x,
#         max_imfs=max_imfs,
#         nensembles=nensembles,
#         nprocesses=nprocesses,      # now guaranteed int
#         ensemble_noise=ensemble_noise,
#         imf_opts=imf_opts
#     )

#     imf = ensure_imf_shape(imf, n_samples_expected=x.size)

#     if drop_res:
#         imf = drop_last_if_residual(imf, fs)

#     return imf



# def match_imfs_by_median_if(imf1, imf2, fs, tol_hz, band=None):
#     """
#     Compute IF for both IMF sets and match IMFs by closest median IF,
#     enforcing 1-1 pairing within a given tolerance.

#     Args:
#         imf1, imf2: arrays (n_imfs, n_samples)
#         fs: sampling rate
#         tol_hz: max allowed difference between median IFs
#         band: optional (fmin, fmax) to restrict matching to a band of interest

#     Returns:
#         pairs: list of (i, j) indices
#         med_if1, med_if2: arrays of median IFs (Hz)
#     """
#     _, IF1, _ = emd.spectra.frequency_transform(imf1, fs, 'hilbert')
#     _, IF2, _ = emd.spectra.frequency_transform(imf2, fs, 'hilbert')

#     med_if1 = np.nanmedian(IF1, axis=1)
#     med_if2 = np.nanmedian(IF2, axis=1)

#     idx1 = np.arange(med_if1.size)
#     idx2 = np.arange(med_if2.size)

#     if band is not None:
#         fmin, fmax = band
#         mask1 = (med_if1 >= fmin) & (med_if1 <= fmax)
#         mask2 = (med_if2 >= fmin) & (med_if2 <= fmax)
#         idx1 = idx1[mask1]
#         idx2 = idx2[mask2]

#     # Build candidate pairs (greedy by nearest IF2)
#     raw_pairs = []
#     for i in idx1:
#         j = idx2[np.argmin(np.abs(med_if2[idx2] - med_if1[i]))] if idx2.size else None
#         if j is not None and np.abs(med_if2[j] - med_if1[i]) <= tol_hz:
#             raw_pairs.append((i, j))

#     # Enforce 1-1 pairing by keeping the closest first
#     pairs = []
#     used_j = set()
#     for i, j in sorted(raw_pairs, key=lambda p: abs(med_if1[p[0]] - med_if2[p[1]])):
#         if j not in used_j:
#             pairs.append((i, j))
#             used_j.add(j)

#     return pairs, med_if1, med_if2


# def mix_by_pairs(imf1, imf2, pairs, n_samples):
#     """
#     Sum matched IMFs into a single time-domain mixed signal.
#     """
#     mixed_components = []
#     for i, j in pairs:
#         comp = imf1[i, :] + imf2[j, :]
#         mixed_components.append(comp)

#     if mixed_components:
#         mixed = np.sum(np.vstack(mixed_components), axis=0)
#     else:
#         mixed = np.zeros(n_samples, dtype=float)

#     return mixed


# # -----------------------
# # Main API
# # -----------------------
# def superposition(
#     s1, s2, fs=42000,
#     IF_TOL_HZ=2000.0,
#     DROP_RESIDUAL=True,
#     BAND=(5.0, 6000.0),  # e.g., (5.0, 5000.0) to restrict matching
#     # EEMD parameters:
#     MAX_IMFS=5,
#     NENSEMBLES=8,
#     ENSEMBLE_NOISE=0.2,
#     NPROCESSES=None,
#     IMF_OPTS=None,
# ):   
    
#     t=np.linspace(0, 1, 2048, endpoint=False)
#     assert s1.shape == s2.shape == t.shape, "s1, s2, t must have the same shape"

#     # 1) EEMD decomposition for both signals
#     imf1 = get_imfs_noise_assisted(
#         s1, fs,
#         drop_res=DROP_RESIDUAL,
#         max_imfs=MAX_IMFS,
#         nensembles=NENSEMBLES,
#         ensemble_noise=ENSEMBLE_NOISE,
#         nprocesses=NPROCESSES,
#         imf_opts=IMF_OPTS
#     )
#     imf2 = get_imfs_noise_assisted(
#         s2, fs,
#         drop_res=DROP_RESIDUAL,
#         max_imfs=MAX_IMFS,
#         nensembles=NENSEMBLES,
#         ensemble_noise=ENSEMBLE_NOISE,
#         nprocesses=NPROCESSES,
#         imf_opts=IMF_OPTS
#     )

#     # 2) Match IMFs by median IF (optionally restricted to a band)
#     pairs, med_if1, med_if2 = match_imfs_by_median_if(
#         imf1, imf2, fs, tol_hz=IF_TOL_HZ, band=BAND
#     )

#     # 3) Mix only matched IMFs
#     mixed = mix_by_pairs(imf1, imf2, pairs, n_samples=t.size)

#     return mixed

def _rms(x, eps=1e-12):
    """Root-mean-square of a 1D signal."""
    x = np.asarray(x, dtype=np.float32).ravel()
    return float(np.sqrt(np.mean(x * x) + eps))

def superposition_fft(s1, s2, target_len=None, normalize="rms"):
    """
    Simple frequency-domain mix:
      - align lengths
      - average magnitudes, keep phase of s1
      - IFFT back
      - optional RMS normalization
    """
    x1 = np.asarray(s1, dtype=np.float32).ravel()
    x2 = np.asarray(s2, dtype=np.float32).ravel()

    if target_len is None:
        L = min(x1.size, x2.size)
    else:
        L = int(target_len)
    if x1.size < L:
        x1 = np.pad(x1, (0, L - x1.size), mode="constant")
    if x2.size < L:
        x2 = np.pad(x2, (0, L - x2.size), mode="constant")
    x1 = x1[:L] - np.mean(x1)
    x2 = x2[:L] - np.mean(x2)

    X1 = np.fft.rfft(x1)
    X2 = np.fft.rfft(x2)

    mag = 0.5 * (np.abs(X1) + np.abs(X2))
    phase = np.angle(X1)
    Y = mag * np.exp(1j * phase)

    mixed = np.fft.irfft(Y, n=L).astype(np.float32)

    if normalize == "rms":
        target_rms = 0.5 * (_rms(x1) + _rms(x2))
        mixed_rms = _rms(mixed)      
        if mixed_rms > 0:
            mixed *= (target_rms / mixed_rms)
        
    return mixed
