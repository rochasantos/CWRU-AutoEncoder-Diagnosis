import torch
import torch.nn as nn
import torch.nn.functional as F

class MCNN_LSTM(nn.Module):
    """
    Multi-scale CNN + LSTM (MCNN-LSTM) as in the paper/figure:
      - Two 1D-CNN branches with different receptive fields/strides
      - Feature fusion by channel concatenation
      - Two stacked LSTMs
      - Final dense layer to num_classes

    Expected input: (batch, 1, 250)  # 1D vibration segment with 250 samples
    Outputs logits (use CrossEntropyLoss). Call predict_proba() for softmax.
    """

    def __init__(self, num_classes=4):
        super().__init__()

        # ------- Branch CNN_1 (top in the figure) -------
        # Conv_1: 1 -> 50, k=20, s=2, tanh; length 250 -> 116
        self.conv1 = nn.Conv1d(1, 50, kernel_size=20, stride=2, bias=True)
        # Conv_2: 50 -> 30, k=10, s=2, tanh; length 116 -> 54
        self.conv2 = nn.Conv1d(50, 30, kernel_size=10, stride=2, bias=True)
        # Pooling_1: k=2, s=2; length 54 -> 27
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ------- Branch CNN_2 (bottom in the figure) -------
        # Conv_3: 1 -> 50, k=6, s=1, tanh; length 250 -> 245
        self.conv3 = nn.Conv1d(1, 50, kernel_size=6, stride=1, bias=True)
        # Conv_4: 50 -> 40, k=6, s=1, tanh; length 245 -> 240
        self.conv4 = nn.Conv1d(50, 40, kernel_size=6, stride=1, bias=True)
        # Pooling_2: k=2, s=2; length 240 -> 120
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Conv_5: 40 -> 30, k=6, s=1, tanh; length 120 -> 115
        self.conv5 = nn.Conv1d(40, 30, kernel_size=6, stride=1, bias=True)
        # Conv_6: 30 -> 30, k=6, s=2, tanh; length 115 -> 55
        self.conv6 = nn.Conv1d(30, 30, kernel_size=6, stride=2, bias=True)
        # Pooling_3: k=2, s=2; length 55 -> 27
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ------- LSTMs (classifier in the figure) -------
        # After fusion we have (batch, 60, 27). We feed LSTM as (batch, 27, 60)
        self.lstm1 = nn.LSTM(
            input_size=60,  # fused feature size per time step
            hidden_size=60, # LSTM_1 units
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.lstm2 = nn.LSTM(
            input_size=60,  # receives output features from lstm1 at each step
            hidden_size=30, # LSTM_2 units
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Final dense layer to 10 classes (softmax in paper; we return logits)
        self.fc = nn.Linear(30, num_classes)

        # Activations as specified in the table
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (B, 1, 250)
        returns: logits (B, num_classes)
        """
        assert x.dim() == 3 and x.size(1) == 1, "Input must be (B, 1, 250)"

        # ----- CNN_1 -----
        y1 = self.tanh(self.conv1(x))      # (B, 50, 116)
        y1 = self.tanh(self.conv2(y1))     # (B, 30, 54)
        y1 = self.pool1(y1)                # (B, 30, 27)

        # ----- CNN_2 -----
        y2 = self.tanh(self.conv3(x))      # (B, 50, 245)
        y2 = self.tanh(self.conv4(y2))     # (B, 40, 240)
        y2 = self.pool2(y2)                # (B, 40, 120)
        y2 = self.tanh(self.conv5(y2))     # (B, 30, 115)
        y2 = self.tanh(self.conv6(y2))     # (B, 30, 55)
        y2 = self.pool3(y2)                # (B, 30, 27)

        # ----- Feature fusion (channel-wise concat) -----
        f = torch.cat([y1, y2], dim=1)     # (B, 60, 27)

        # ----- Prepare for LSTMs: (B, T=27, F=60) -----
        f = f.transpose(1, 2)              # (B, 27, 60)

        # LSTM_1 + ReLU on outputs (as in the table)
        out1, _ = self.lstm1(f)            # (B, 27, 60)
        out1 = self.relu(out1)

        # LSTM_2
        out2, _ = self.lstm2(out1)         # (B, 27, 30)

        # Use last time step (the paper's table shows 1×30 before Dense)
        last = out2[:, -1, :]              # (B, 30)

        logits = self.fc(last)             # (B, num_classes)
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        """Return class probabilities."""
        return F.softmax(self.forward(x), dim=-1)
