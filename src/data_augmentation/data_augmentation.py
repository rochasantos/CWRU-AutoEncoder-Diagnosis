import numpy as np
import random
import torch

# --------- suas transforms (copie as que você já tem aqui) ----------
fc = 42000
f_o = 107.36
f_i = 162.19
f_b = 70.58

def add_gaussian_noise(signal, snr_db=10):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise

def local_data_reversing(signal, M=10):
    N = len(signal)
    min_len = int(fc / min(f_o, f_i, f_b))
    segment_len = max(N // M, min_len)
    segments = np.array_split(signal[:M * segment_len].copy(), M)
    for i in range(len(segments)):
        sub_len = np.random.randint(1, segment_len)
        start = np.random.randint(0, max(1, segment_len - sub_len))
        segments[i][start:start+sub_len] = segments[i][start:start+sub_len][::-1]
    return np.concatenate(segments)

def local_random_reversing(signal, q=0.2):
    sig = signal.copy()
    N = len(sig)
    min_len = int(fc / min(f_o, f_i, f_b))
    seg_len = max(int(q * N), min_len)
    if seg_len >= N:
        return sig[::-1]
    start = np.random.randint(0, N - seg_len)
    sig[start:start+seg_len] = sig[start:start+seg_len][::-1]
    return sig

def global_data_reversing(signal):
    return signal[::-1].copy()

def local_data_zooming(signal, q=0.2, nmin=0.4, nmax=1.6):
    N = len(signal)
    seg_len = max(1, int(q * N))
    seg_len = min(seg_len, N)
    segment = signal[:seg_len]
    idxs = sorted(np.random.randint(seg_len, max(seg_len+1, N - seg_len + 1), size=3))
    new_signal = signal.copy()
    for idx in idxs:
        factor = np.random.uniform(nmin, nmax)
        end = min(idx+seg_len, N)
        new_signal[idx:end] = factor * segment[:end-idx]
    return new_signal

def global_data_zooming(signal, nmin=0.4, nmax=1.6):
    factor = np.random.uniform(nmin, nmax)
    return signal * factor

def local_segment_splicing(signal, M=20):
    N = len(signal)
    if M <= 1:
        return signal.copy()
    segment_len = max(1, N // M)
    segments = np.array_split(signal[:M * segment_len].copy(), M)
    np.random.shuffle(segments)
    tail = signal[M * segment_len:]  # keep tail if exists
    return np.concatenate(segments + ([tail] if tail.size else []))

def permute_signal(signal, n_segments=8, seed=None):
    if seed is not None:
        np.random.seed(seed)
    N = len(signal)
    n_segments = max(1, min(n_segments, N))
    # make length divisible by n_segments by trimming small tail if necessary
    L = (N // n_segments) * n_segments
    core = signal[:L].copy()
    segments = np.array_split(core, n_segments)
    permuted_indices = np.random.permutation(n_segments)
    permuted_segments = [segments[i] for i in permuted_indices]
    mixed = np.concatenate(permuted_segments)
    tail = signal[L:]
    return np.concatenate([mixed, tail]) if tail.size else mixed
# ---------------------------------------------------------------------

class _AugmentedDataset:
    """
    Wrap any Dataset and apply one random augmentation to the signal on each __getitem__.
    - Preserves original shape (e.g., (L,), (1, L), (L, 1)).
    - Works with datasets returning (x, y), [x, y], or dicts (keys: 'signal', 'x', 'data', 'input').
    """
    def __init__(self, base_dataset, transforms=None, p=1.0, enabled=True, rng_seed=None):
        self.base = base_dataset
        self.enabled = enabled
        self.p = float(p)
        if transforms is None:
            transforms = [
                add_gaussian_noise,
                local_data_reversing,
                local_random_reversing,
                global_data_reversing,
                local_data_zooming,
                global_data_zooming,
                local_segment_splicing,
                permute_signal,
            ]
        self.transforms = transforms
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

    def __len__(self):
        return len(self.base)

    # ---------- helpers: shape & dtype/device preservation ----------
    def _flatten_to_1d_numpy(self, x):
        """
        Return (arr_1d, meta)
        meta carries enough info to rebuild original: ('torch'|'numpy', dtype, device_or_None, orig_shape)
        """
        if isinstance(x, torch.Tensor):
            orig_shape = tuple(x.shape)
            arr = x.detach().cpu().contiguous().view(-1).numpy().copy()
            return arr, ('torch', x.dtype, x.device, orig_shape)
        elif isinstance(x, np.ndarray):
            orig_shape = tuple(x.shape)
            arr = np.ascontiguousarray(x.reshape(-1)).copy()
            return arr, ('numpy', x.dtype, None, orig_shape)
        else:
            arr = np.asarray(x).reshape(-1).copy()
            return arr, ('coerced', arr.dtype, None, (len(arr),))

    def _restore_from_1d_numpy(self, arr_1d, meta):
        kind, dtype, device, orig_shape = meta
        # reshape back to original
        out_np = np.asarray(arr_1d).reshape(orig_shape)
        if kind == 'torch':
            return torch.as_tensor(out_np, dtype=dtype, device=device)
        elif kind == 'numpy':
            return out_np.astype(dtype, copy=False)
        else:
            return out_np  # best effort

    def _pick_and_apply(self, sig_np_1d):
        aug = random.choice(self.transforms)
        try:
            out = aug(sig_np_1d)
            return np.asarray(out).reshape(-1)
        except Exception:
            return sig_np_1d

    def _augment_signal_if_needed(self, x):
        if not self.enabled or random.random() > self.p:
            return x
        sig_1d, meta = self._flatten_to_1d_numpy(x)
        aug_1d = self._pick_and_apply(sig_1d)
        return self._restore_from_1d_numpy(aug_1d, meta)
    # ----------------------------------------------------------------

    def __getitem__(self, idx):
        item = self.base[idx]

        if isinstance(item, (tuple, list)) and len(item) >= 1:
            x = item[0]
            x_aug = self._augment_signal_if_needed(x)
            if isinstance(item, tuple):
                return (x_aug,) + tuple(item[1:])
            out = list(item)
            out[0] = x_aug
            return out

        if isinstance(item, dict):
            for k in ['signal', 'x', 'data', 'input']:
                if k in item:
                    new_item = dict(item)
                    new_item[k] = self._augment_signal_if_needed(item[k])
                    return new_item
            return item

        return self._augment_signal_if_needed(item)



def wrap_with_random_augmentation(dataset, transforms=None, p=1.0, enabled=True, rng_seed=None):
    """
    Public helper: wrap your existing dataset with on-the-fly random augmentation.
    - dataset: your current Dataset instance (sem refatorar!)
    - transforms: lista de funções (cada uma recebe e retorna np.ndarray 1D)
    - p: probabilidade de aplicar alguma transformação (0..1)
    - enabled: liga/desliga rapidamente
    - rng_seed: para reprodutibilidade
    """
    return _AugmentedDataset(dataset, transforms=transforms, p=p, enabled=enabled, rng_seed=rng_seed)
