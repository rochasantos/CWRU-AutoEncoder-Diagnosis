import torch
import torch.nn as nn
import torch.nn.functional as F

class MCNN_LSTM_LowFreq(nn.Module):
    """
    Slim MCNN-LSTM that keeps ONLY the low-frequency branch from the paper:
      Conv_1 (k=20,s=2) -> Conv_2 (k=10,s=2) -> Pooling_1 (k=2,s=2)
    Then: LSTM -> ReLU -> LSTM -> Dense
    - Input expected: (B, 1, 250)
    - Output: logits (B, num_classes). Use CrossEntropyLoss.
    - If num_classes == 2, this is a binary classifier (2 logits).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # -------- low-frequency CNN path (Conv_1 branch) --------
        # Conv_1: 1 -> 50, kernel=20, stride=2, tanh; length 250 -> 116
        self.conv1 = nn.Conv1d(1, 50, kernel_size=8, stride=2, bias=True)
        # Conv_2: 50 -> 30, kernel=10, stride=2, tanh; length 116 -> 54
        self.conv2 = nn.Conv1d(50, 30, kernel_size=4, stride=2, bias=True)
        # Pooling_1: kernel=2, stride=2; length 54 -> 27
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # -------- LSTMs (same spirit as paper, adapted input size=30) --------
        # After Pooling_1 we have (B, 30, 27) -> transpose to (B, 27, 30)
        self.lstm1 = nn.LSTM(input_size=30, hidden_size=60, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=60, hidden_size=30, num_layers=1, batch_first=True)

        # -------- Classifier --------
        self.fc = nn.Linear(30, num_classes)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (B, 1, 250)
        returns: logits (B, num_classes)
        """
        assert x.dim() == 3 and x.size(1) == 1, "Input must be (B, 1, 250)"

        # Low-frequency CNN path
        y = self.tanh(self.conv1(x))    # (B, 50, 116)
        y = self.tanh(self.conv2(y))    # (B, 30, 54)
        y = self.pool1(y)               # (B, 30, 27)

        # LSTMs
        y = y.transpose(1, 2)           # (B, 27, 30)
        y, _ = self.lstm1(y)            # (B, 27, 60)
        y = self.relu(y)
        y, _ = self.lstm2(y)            # (B, 27, 30)

        last = y[:, -1, :]              # (B, 30)
        logits = self.fc(last)          # (B, num_classes)
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        """Return softmax probabilities. If num_classes==2 -> binary probs [p0, p1]."""
        return F.softmax(self.forward(x), dim=-1)
