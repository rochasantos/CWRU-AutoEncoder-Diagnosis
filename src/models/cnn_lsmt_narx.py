import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CNN backbone that outputs one feature vector per window
class CNN1DBackbone(nn.Module):
    def __init__(self, in_channels=1, out_channels=(16, 32), kernels=(16, 32)):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels[0], kernels[0], padding=kernels[0]//2)
        self.bn1   = nn.BatchNorm1d(out_channels[0])
        self.pool1 = nn.MaxPool1d(8)

        self.conv2 = nn.Conv1d(out_channels[0], out_channels[1], kernels[1], padding=kernels[1]//2)
        self.bn2   = nn.BatchNorm1d(out_channels[1])
        self.pool2 = nn.MaxPool1d(8)

        self.gap   = nn.AdaptiveAvgPool1d(1)  # feature per window

    def forward(self, x):
        # x: [B, 1, Lw]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.gap(x).squeeze(-1)           # [B, C2]
        return x

# --- CNN + LSTM (no NARX)
class CNNLSTMCompat(nn.Module):
    """
    Drop-in replacement for BearingCNN1D:
      - forward(x) receives raw signal [B, 1, L]
      - internally windows the signal and applies CNN -> LSTM -> classifier
    """
    def __init__(self, num_classes, in_channels=1, out_channels=(16, 32), kernels=(16, 32),
                 window_size=2048, hop=512, lstm_hidden=64, lstm_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.window_size = window_size
        self.hop = hop

        self.backbone = CNN1DBackbone(in_channels, out_channels, kernels)

        feat_dim = out_channels[1]
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, num_classes)

    def make_windows(self, x):
        # x: [B, 1, L] -> [B, T, 1, window_size]
        B, C, L = x.shape
        if L < self.window_size:
            # Right-pad with zeros so we always have at least one window
            pad_len = self.window_size - L
            x = F.pad(x, (0, pad_len))
            L = self.window_size

        T = 1 + (L - self.window_size) // self.hop
        if T <= 0:
            T = 1

        xs = []
        for t in range(T):
            xs.append(x[:, :, t*self.hop:t*self.hop+self.window_size])
        return torch.stack(xs, dim=1)  # [B, T, 1, window_size]

    def forward(self, x):
        # x: [B, 1, L]
        xw = self.make_windows(x)          # [B, T, 1, Lw]
        B, T, C, Lw = xw.shape

        # Extract CNN features per window
        feats = []
        for t in range(T):
            z_t = self.backbone(xw[:, t])  # [B, D]
            feats.append(z_t)
        z_seq = torch.stack(feats, dim=1)  # [B, T, D]

        # LSTM over feature sequence
        out, _ = self.lstm(z_seq)          # [B, T, H]
        h_last = out[:, -1]                # last time step
        logits = self.head(h_last)         # [B, num_classes]
        return logits
