import torch
import torch.nn as nn
import torch.nn.functional as F

class MCNN_LSTM(nn.Module):
    """
    Multi-scale CNN + LSTM that works with any input length L.
    Input:  (B, 1, L)  with arbitrary L
    Output: logits (B, num_classes)
    Key changes:
      - Added padding to all convs so very short sequences won't crash.
      - AdaptiveAvgPool1d aligns both branches to the same temporal T.
    """

    def __init__(self, num_classes=4, target_T=27):
        super().__init__()

        # ------- Branch CNN_1 -------
        # pad = k//2 to tolerate small L
        self.conv1 = nn.Conv1d(1, 50, kernel_size=20, stride=2, padding=10, bias=True)
        self.conv2 = nn.Conv1d(50, 30, kernel_size=10, stride=2, padding=5,  bias=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ------- Branch CNN_2 -------
        self.conv3 = nn.Conv1d(1, 50, kernel_size=6, stride=1, padding=3, bias=True)
        self.conv4 = nn.Conv1d(50, 40, kernel_size=6, stride=1, padding=3, bias=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(40, 30, kernel_size=6, stride=1, padding=3, bias=True)
        self.conv6 = nn.Conv1d(30, 30, kernel_size=6, stride=2, padding=3, bias=True)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Align both branches to the same temporal length T
        self.align1 = nn.AdaptiveAvgPool1d(target_T)
        self.align2 = nn.AdaptiveAvgPool1d(target_T)

        # ------- LSTMs -------
        self.lstm1 = nn.LSTM(input_size=60, hidden_size=60, num_layers=1,
                             batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=60, hidden_size=30, num_layers=1,
                             batch_first=True, bidirectional=False)

        self.fc = nn.Linear(30, num_classes)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, 1, L) with any L>=1
        assert x.dim() == 3 and x.size(1) == 1, "Input must be (B, 1, L)"

        # ----- CNN_1 -----
        y1 = self.tanh(self.conv1(x))
        y1 = self.tanh(self.conv2(y1))
        y1 = self.pool1(y1)

        # ----- CNN_2 -----
        y2 = self.tanh(self.conv3(x))
        y2 = self.tanh(self.conv4(y2))
        y2 = self.pool2(y2)
        y2 = self.tanh(self.conv5(y2))
        y2 = self.tanh(self.conv6(y2))
        y2 = self.pool3(y2)

        # ----- Temporal alignment -----
        y1 = self.align1(y1)             # (B, 30, T)
        y2 = self.align2(y2)             # (B, 30, T)

        # ----- Fusion -----
        f = torch.cat([y1, y2], dim=1)   # (B, 60, T)

        # ----- LSTMs -----
        f = f.transpose(1, 2)            # (B, T, 60)
        out1, _ = self.lstm1(f)          # (B, T, 60)
        out1 = self.relu(out1)
        out2, _ = self.lstm2(out1)       # (B, T, 30)

        last = out2[:, -1, :]            # (B, 30)
        logits = self.fc(last)           # (B, num_classes)
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=-1)
