import torch
import torch.nn as nn
import torch.nn.functional as F

class HPSO_CNN_LSTM(nn.Module):
    def __init__(self, input_size=(4096, 1, 1), num_classes=10):
        super(HPSO_CNN_LSTM, self).__init__()

        # CNN parameters
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=2)  # Número de filtros = 6
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=6, kernel_size=2)  # Número de filtros = 6

        self.relu = nn.ReLU()  # Activation = 'relu'

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Maxpooling, pooling length=2, stride=2

        # LSTM parameters
        self.lstm_input_size = 6  # Output channels from CNN
        self.lstm_hidden_size = 197  # Number of hidden layer units = 197
        self.num_lstm_layers = 2

        # LSTM input dimension: precisamos calcular após flatten
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)

        # Output layer
        self.fc = nn.Linear(self.lstm_hidden_size, num_classes)

        # Softmax classifier
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels, width)
        x = x.view(x.size(0), 1, -1)  # reshape for Conv1d: (batch_size, channels=1, seq_len)

        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Prepare for LSTM: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Pegamos a saída do último time step
        out = lstm_out[:, -1, :]

        out = self.fc(out)

        out = self.softmax(out)

        return out

