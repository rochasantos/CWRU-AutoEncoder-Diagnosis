import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import hilbert

def compute_hilbert_envelope(signal):
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope
