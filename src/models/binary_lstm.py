import torch
import torch.nn as nn

class BinaryLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=197, num_layers=2, num_classes=1):
        """
        Args:
            input_size: número de features por timestep (para sinais de vibração, normalmente = 1).
            hidden_size: número de neurônios por camada LSTM.
            num_layers: número de camadas LSTM.
            num_classes: saída -> 1 para classificação binária com BCEWithLogitsLoss.
        """
        super(BinaryLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: shape (batch_size, seq_len, input_size)
        """
        lstm_out, _ = self.lstm(x)
        # Pega o último timestep
        out = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size)

        out = self.fc(out)  # shape: (batch_size, num_classes)

        return out
