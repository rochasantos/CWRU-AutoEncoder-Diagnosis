import random
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

class BalancedBatchSampler:
    """
    Yields balanced batches: same number of samples per class in each batch.
    - labels: list/array with the class index for each sample (len == len(dataset))
    - batch_size must be divisible by n_classes
    - Oversamples minority classes when needed
    """
    def __init__(self, labels, batch_size, shuffle=True, drop_last=False):
        self.labels = list(labels)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        # Build index lists per class
        self.class_to_indices = defaultdict(list)
        for idx, y in enumerate(self.labels):
            self.class_to_indices[int(y)].append(idx)

        self.classes = sorted(self.class_to_indices.keys())
        self.n_classes = len(self.classes)
        assert self.batch_size % self.n_classes == 0, \
            f"batch_size ({self.batch_size}) must be divisible by n_classes ({self.n_classes})"

        self.samples_per_class = self.batch_size // self.n_classes

        # Total "epoch length" heuristic: cover roughly one pass over all samples
        self.total_size = len(self.labels)
        # Number of batches we aim to produce in an epoch
        self.num_batches = (self.total_size // self.batch_size) if self.drop_last else -(-self.total_size // self.batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # Prepare per-class pools (copy and shuffle each class list)
        pools = {}
        for c in self.classes:
            indices = self.class_to_indices[c][:]
            if self.shuffle:
                random.shuffle(indices)
            pools[c] = indices

        # Pointers per class
        ptr = {c: 0 for c in self.classes}

        # Helper to take k indices from class c, with oversampling if needed
        def take_from_class(c, k):
            out = []
            while len(out) < k:
                remaining = len(pools[c]) - ptr[c]
                if remaining <= 0:
                    # Refill (oversample) by reshuffling the original class pool
                    new_block = self.class_to_indices[c][:]
                    if self.shuffle:
                        random.shuffle(new_block)
                    pools[c].extend(new_block)
                    remaining = len(pools[c]) - ptr[c]
                take = min(k - len(out), remaining)
                out.extend(pools[c][ptr[c]:ptr[c]+take])
                ptr[c] += take
            return out

        # Yield balanced batches
        produced = 0
        while produced < self.num_batches:
            batch = []
            for c in self.classes:
                batch.extend(take_from_class(c, self.samples_per_class))
            # Optional shuffle inside the batch
            if self.shuffle:
                random.shuffle(batch)
            yield batch
            produced += 1