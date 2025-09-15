import numpy as np

fc = 42000  # Frequência de amostragem típica no CWRU (pode ser ajustada)
f_o = 107.36  # BPFO em Hz
f_i = 162.19  # BPFI em Hz
f_b = 70.58 

# Noise
def add_gaussian_noise(signal, snr_db=10):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise

# Local Reversing
def local_data_reversing(signal, M=10):
        N = len(signal)
        min_len = int(fc / min(f_o, f_i, f_b))
        segment_len = max(N // M, min_len)
        segments = np.array_split(signal[:M * segment_len], M)
        for i in range(len(segments)):
            sub_len = np.random.randint(1, segment_len)
            start = np.random.randint(0, segment_len - sub_len)
            segments[i][start:start+sub_len] = segments[i][start:start+sub_len][::-1]
        return np.concatenate(segments)


def local_random_reversing(signal, q=0.2):
    N = len(signal)
    min_len = int(fc / min(f_o, f_i, f_b))
    seg_len = max(int(q * N), min_len)
    start = np.random.randint(0, N - seg_len)
    signal[start:start+seg_len] = signal[start:start+seg_len][::-1]
    return signal

def global_data_reversing(signal):
    return signal[::-1]

def local_data_zooming(signal, q=0.2, nmin=0.4, nmax=1.6):
    N = len(signal)
    seg_len = int(q * N)
    segment = signal[:seg_len]
    indexes = sorted(np.random.randint(seg_len, N - seg_len, size=3))
    scaled = [(np.random.uniform(nmin, nmax) * segment) for _ in indexes]
    new_signal = signal.copy()
    for idx, scale_seg in zip(indexes, scaled):
        new_signal[idx:idx+seg_len] = scale_seg[:min(seg_len, N - idx)]
    return new_signal

def global_data_zooming(signal, nmin=0.4, nmax=1.6):
    factor = np.random.uniform(nmin, nmax)
    return signal * factor

def local_segment_splicing(signal, M=20):
    N = len(signal)
    min_len = int(fc / min(f_o, f_i, f_b))
    segment_len = max(N // M, min_len)
    segments = np.array_split(signal[:M * segment_len], M)
    np.random.shuffle(segments)
    return np.concatenate(segments)

def permute_signal(signal, n_segments=8, seed=None):        
    if seed is not None:
        np.random.seed(seed)
    if len(signal) % n_segments != 0:
        raise ValueError("Signal length must be divisible by n_segments.")        
    segments = np.array_split(signal, n_segments)
    permuted_indices = np.random.permutation(n_segments)
    permuted_segments = [segments[i] for i in permuted_indices]
    permuted_signal = np.concatenate(permuted_segments)

    return permuted_signal
