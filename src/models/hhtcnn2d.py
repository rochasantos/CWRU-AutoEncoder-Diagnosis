import torch
import torch.nn as nn

import torch

def ensure_5d(x):
    """
    Ensure input is (B, C, D, H, W) for Conv3d.
    Accepts (B,H,W), (B,C,H,W), or (B,D,H,W) and upgrades to 5D.
    """
    if x.dim() == 3:
        # (B, H, W) -> (B, 1, 1, H, W)
        x = x.unsqueeze(1).unsqueeze(2)
    elif x.dim() == 4:
        # Could be (B, C, H, W) or (B, D, H, W)
        B, A, H, W = x.shape
        if A in (1, 3):
            # assume A=channels -> (B, C, 1, H, W)
            x = x.unsqueeze(2)
        else:
            # assume A=depth -> (B, 1, D, H, W)
            x = x.unsqueeze(1)
    elif x.dim() != 5:
        raise ValueError(f"Expected 3D/4D/5D tensor, got {x.shape}")
    return x


class HHTCNN3D(nn.Module):
    def __init__(self, num_classes, in_channels=1, base_channels=16, dropout_p=0.3):
        super().__init__()
        # Block 1 (mantém D, reduz HxW)
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2))  # pool só em H e W
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2))  # ainda evita mexer em D
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 1))  # NÃO encolhe W de 2→0, só reduz H
        )


        # Neck + Head
        self.neck = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # global pooling 3D
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(base_channels*4, num_classes)
        )

    def forward(self, x):
        x = ensure_5d(x)
        # sanity check to fail early with a clear message
        assert x.dim() == 5, f"Got {x.shape}, expected (B,C,D,H,W)"
        # x shape: (B, C, D, H, W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.neck(x)
        x = self.classifier(x)
        return x
