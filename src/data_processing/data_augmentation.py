import numpy as np

class TransformDataAugmentation:
    def __init__(self, prob=0.5, normalize=True):
        # Parâmetros físicos do rolamento 6205 CWRU
        self.fc = 48000  # Frequência de amostragem típica no CWRU (pode ser ajustada)
        self.f_o = 107.36  # BPFO em Hz
        self.f_i = 162.19  # BPFI em Hz
        self.f_b = 70.58  # BSF em Hz
        self.prob = prob
        self.normalize = normalize

    def __call__(self, signal):
        signal = signal.astype(np.float32)

        if self.normalize:
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        if np.random.rand() > self.prob:
            return signal

        return self.apply_random_augmentation(signal)

    def apply_random_augmentation(self, signal):
        methods = [
            lambda x: self.local_data_reversing(x),
            lambda x: self.local_random_reversing(x),
            self.global_data_reversing,
            self.local_data_zooming,
            self.global_data_zooming,
            lambda x: self.local_segment_splicing(x),
            self.add_gaussian_noise
        ]
        method = np.random.choice(methods)
        return method(signal)

    def add_gaussian_noise(self, signal, snr_db=20):
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
        return signal + noise

    def local_data_reversing(self, signal, M=10):
        N = len(signal)
        min_len = int(self.fc / min(self.f_o, self.f_i, self.f_b))
        segment_len = max(N // M, min_len)
        segments = np.array_split(signal[:M * segment_len], M)
        for i in range(len(segments)):
            sub_len = np.random.randint(1, segment_len)
            start = np.random.randint(0, segment_len - sub_len)
            segments[i][start:start+sub_len] = segments[i][start:start+sub_len][::-1]
        return np.concatenate(segments)

    def local_random_reversing(self, signal, q=0.2):
        N = len(signal)
        min_len = int(self.fc / min(self.f_o, self.f_i, self.f_b))
        seg_len = max(int(q * N), min_len)
        start = np.random.randint(0, N - seg_len)
        signal[start:start+seg_len] = signal[start:start+seg_len][::-1]
        return signal

    def global_data_reversing(self, signal):
        return signal[::-1]

    def local_data_zooming(self, signal, q=0.2, nmin=0.4, nmax=1.6):
        N = len(signal)
        seg_len = int(q * N)
        segment = signal[:seg_len]
        indexes = sorted(np.random.randint(seg_len, N - seg_len, size=3))
        scaled = [(np.random.uniform(nmin, nmax) * segment) for _ in indexes]
        new_signal = signal.copy()
        for idx, scale_seg in zip(indexes, scaled):
            new_signal[idx:idx+seg_len] = scale_seg[:min(seg_len, N - idx)]
        return new_signal

    def global_data_zooming(self, signal, nmin=0.4, nmax=1.6):
        factor = np.random.uniform(nmin, nmax)
        return signal * factor

    def local_segment_splicing(self, signal, M=20):
        N = len(signal)
        min_len = int(self.fc / min(self.f_o, self.f_i, self.f_b))
        segment_len = max(N // M, min_len)
        segments = np.array_split(signal[:M * segment_len], M)
        np.random.shuffle(segments)
        return np.concatenate(segments)
