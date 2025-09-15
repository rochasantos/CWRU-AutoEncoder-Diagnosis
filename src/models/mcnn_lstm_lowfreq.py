import torch
import torch.nn as nn
import torch.nn.functional as F

class MCNN_LSTM_LowFreq(nn.Module):
    """
    Low-frequency MCNN-LSTM that works with any input length L.
    Input:  (B, 1, L) with arbitrary L >= 1
    Output: logits (B, num_classes)
    Changes:
      - Convs use padding to tolerate small L.
      - Safe "/2" pooling via adaptive_max_pool1d (never collapses to 0).
      - AdaptiveAvgPool1d aligns time steps to target_T for LSTMs.
    """

    def __init__(self, num_classes=2, target_T=27):
        super().__init__()

        # ---- Low-frequency CNN path ----
        # Using padding=k//2 to be robust for small inputs
        self.conv1 = nn.Conv1d(1, 50, kernel_size=8, stride=2, padding=4, bias=True)
        self.conv2 = nn.Conv1d(50, 30, kernel_size=4, stride=2, padding=2, bias=True)

        # We emulate MaxPool1d(k=2,s=2) safely inside forward.
        # Final alignment to a fixed temporal length:
        self.align = nn.AdaptiveAvgPool1d(target_T)

        # ---- LSTMs ----
        self.lstm1 = nn.LSTM(input_size=30, hidden_size=60, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=60, hidden_size=30, num_layers=1, batch_first=True)

        # ---- Classifier ----
        self.fc = nn.Linear(30, num_classes)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def _safe_half_pool(self, y):
        """
        Emulates a stride-2 pooling safely. If time length T is very small,
        we still return at least 1 time step.
        """
        T = y.size(-1)
        out_T = max(1, T // 2)  # target ~ T/2, but never 0
        return F.adaptive_max_pool1d(y, out_T)

    def forward(self, x):
        # x: (B, 1, L) with any L >= 1
        assert x.dim() == 3 and x.size(1) == 1, "Input must be (B, 1, L)"

        # Low-frequency CNN path
        y = self.tanh(self.conv1(x))    # (B, 50, ~L/2)
        y = self.tanh(self.conv2(y))    # (B, 30, ~L/4)
        y = self._safe_half_pool(y)     # (B, 30, ~L/8), but >= 1

        # Align to fixed temporal steps for LSTMs
        y = self.align(y)               # (B, 30, target_T)

        # LSTMs
        y = y.transpose(1, 2)           # (B, target_T, 30)
        y, _ = self.lstm1(y)            # (B, target_T, 60)
        y = self.relu(y)
        y, _ = self.lstm2(y)            # (B, target_T, 30)

        last = y[:, -1, :]              # (B, 30)
        logits = self.fc(last)          # (B, num_classes)
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=-1)
