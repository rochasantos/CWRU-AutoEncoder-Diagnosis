import numpy as np

class DataAugmentation:
    
    @staticmethod
    def signal_shuffle(signal, segment_size):
        num_segments = len(signal) // segment_size
        segments = np.array_split(signal[:num_segments * segment_size], num_segments)
        np.random.shuffle(segments)
        augmented_signal = np.concatenate(segments)
        return augmented_signal

    @staticmethod
    def mix_signals(signal1, signal2, segment_size):
        num_segments = min(len(signal1), len(signal2)) // segment_size
        segments1 = np.array_split(signal1[:num_segments * segment_size], num_segments)
        segments2 = np.array_split(signal2[:num_segments * segment_size], num_segments)
        
        mixed_signal1 = np.concatenate([segments1[i] if i % 2 == 0 else segments2[i] for i in range(num_segments)])
        mixed_signal2 = np.concatenate([segments2[i] if i % 2 == 0 else segments1[i] for i in range(num_segments)])
        
        return mixed_signal1, mixed_signal2