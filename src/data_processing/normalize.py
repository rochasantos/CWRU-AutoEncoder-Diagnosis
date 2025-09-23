import numpy as np
import torch

def normalize_signal(x, mode="zscore"):
    """
    Normalize a 1D numpy array or torch tensor.
    
    Args:
        x: numpy array or torch tensor with shape (L,)
        mode: 'zscore', 'minmax', or 'rms'
    
    Returns:
        Normalized tensor (float32)
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    else:
        x = x.to(torch.float32)

    if mode == "zscore":
        mean = x.mean()
        std = x.std()
        if std < 1e-8:  # evita divisão por zero
            return x - mean
        return (x - mean) / std

    elif mode == "minmax":
        min_val = x.min()
        max_val = x.max()
        if (max_val - min_val) < 1e-8:
            return x - min_val
        return (x - min_val) / (max_val - min_val)

    elif mode == "rms":
        rms = torch.sqrt(torch.mean(x**2))
        if rms < 1e-8:
            return x
        return x / rms
    elif mode == "none":
        return x
    else:
        raise ValueError(f"Unknown mode {mode}")
