import torch
import torch.nn as nn

class BearingCNN1D(nn.Module):
    def __init__(self, num_classes, in_channels=1, out_channels=(16, 32), kernels=(16, 32), dropout_p=0.4):
        super().__init__()
        # --- First block ---
        self.conv1 = nn.Conv1d(in_channels, out_channels[0], kernels[0], padding=kernels[0]//2)
        self.bn1   = nn.BatchNorm1d(out_channels[0])
        self.act1  = nn.ReLU()
        self.pool1 = nn.MaxPool1d(8)
        self.drop1 = nn.Dropout(p=dropout_p)

        # --- Second block ---
        self.conv2 = nn.Conv1d(out_channels[0], out_channels[1], kernels[1], padding=kernels[1]//2)
        self.bn2   = nn.BatchNorm1d(out_channels[1])
        self.act2  = nn.ReLU()
        self.pool2 = nn.MaxPool1d(8)
        self.drop2 = nn.Dropout(p=dropout_p)

        # --- Third block ---
        self.conv3 = nn.Conv1d(out_channels[1], out_channels[0], kernels[0], padding=kernels[1]//2)
        self.bn3   = nn.BatchNorm1d(out_channels[0])
        self.act3  = nn.ReLU()
        self.pool3 = nn.MaxPool1d(8)
        self.drop3 = nn.Dropout(p=dropout_p)

        # --- Classifier head ---
        self.head  = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channels[0], num_classes)
        )

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Head
        x = self.head(x)
        return x
