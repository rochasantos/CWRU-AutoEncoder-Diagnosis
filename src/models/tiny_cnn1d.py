# tiny_models_1d.py
import torch
import torch.nn as nn

class TinyCNN1D(nn.Module):
    """
    Compact CNN for very short 1D signals (e.g., length = 26).
    - No downsampling/pooling to avoid zero-sized outputs.
    - Uses dilated 3x convs to enlarge receptive field while preserving length.
    - InstanceNorm works well with small batch sizes.
    - Head uses AdaptiveAvgPool1d(1) to be length-agnostic.
    Expected input shape: [batch, 1, 26]
    """
    def __init__(self, num_classes, in_channels=1, c1=32, c2=64, c3=64, p_drop=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=3, padding=1, dilation=1),
            nn.InstanceNorm1d(c1),
            nn.GELU(),

            nn.Conv1d(c1, c2, kernel_size=8, padding=2, dilation=2),
            nn.InstanceNorm1d(c2),
            nn.GELU(),

            nn.Conv1d(c2, c3, kernel_size=3, padding=4, dilation=4),
            nn.InstanceNorm1d(c3),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # [B, C, L] -> [B, C, 1]
            nn.Flatten(),              # [B, C]
            nn.Dropout(p_drop),
            nn.Linear(c3, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, 26]
        x = self.features(x)
        x = self.head(x)
        return x
