import torch
import torch.nn as nn
import torch.nn.functional as F

class BearingCNN1D(nn.Module):
    """
    Three-block 1D CNN with a stronger classifier head.
    - forward() returns logits (use with CrossEntropyLoss).
    - predict_proba() returns softmax probabilities.
    """
    def __init__(
        self,
        num_classes,
        in_channels=1,
        out_channels=(8, 16, 12),
        kernels=(4, 8, 4),
        dropout_p=0.4,
        hidden=64
    ):
        super().__init__()

        assert len(out_channels) == 3 and len(kernels) == 3, \
            "out_channels and kernels must have length 3 for the 3 conv blocks."

        # --- Feature extractor ---
        # Block 1
        self.conv1 = nn.Conv1d(in_channels, out_channels[0], kernels[0], padding=kernels[0]//2)
        self.bn1   = nn.BatchNorm1d(out_channels[0])
        self.act1  = nn.ReLU()
        self.pool1 = nn.MaxPool1d(8)
        self.drop1 = nn.Dropout(p=dropout_p)

        # Block 2
        self.conv2 = nn.Conv1d(out_channels[0], out_channels[1], kernels[1], padding=kernels[1]//2)
        self.bn2   = nn.BatchNorm1d(out_channels[1])
        self.act2  = nn.ReLU()
        self.pool2 = nn.MaxPool1d(8)
        self.drop2 = nn.Dropout(p=dropout_p)

        # Block 3 (novo)
        self.conv3 = nn.Conv1d(out_channels[1], out_channels[2], kernels[2], padding=kernels[2]//2)
        self.bn3   = nn.BatchNorm1d(out_channels[2])
        self.act3  = nn.ReLU()
        self.pool3 = nn.MaxPool1d(8)
        self.drop3 = nn.Dropout(p=dropout_p)

        # --- Classifier (antes "head") ---
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),          # [B, C, L] -> [B, C, 1]
            nn.Flatten(),                     # [B, C, 1] -> [B, C]
            nn.LayerNorm(out_channels[2]),
            nn.Dropout(p=dropout_p),
            nn.Linear(out_channels[2], hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, num_classes)     # logits
        )

    def forward(self, x):
        # Block 1
        x = self.drop1(self.pool1(self.act1(self.bn1(self.conv1(x)))))
        # Block 2
        x = self.drop2(self.pool2(self.act2(self.bn2(self.conv2(x)))))
        # Block 3
        x = self.drop3(self.pool3(self.act3(self.bn3(self.conv3(x)))))
        # Classifier
        logits = self.classifier(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return F.softmax(self.forward(x), dim=1)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return torch.argmax(self.forward(x), dim=1)
