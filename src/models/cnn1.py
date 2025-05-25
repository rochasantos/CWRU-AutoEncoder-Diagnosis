import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryCNN(nn.Module):
    def __init__(self, input_channels=1, input_length=9600):
        """
        Args:
            input_channels (int): number of input channels.
            input_length (int): length of the input signal.
        """
        super(BinaryCNN, self).__init__()
        self.input_channels = input_channels
        self.input_length = input_length

        self.ln = nn.LayerNorm(input_length)
        self.conv1 = nn.Conv1d(input_channels, out_channels=16, kernel_size=16, padding=1)
        self.pool1 = nn.MaxPool1d(8)

        self.conv2 = nn.Conv1d(16, out_channels=32, kernel_size=32, padding=1)
        self.pool2 = nn.MaxPool1d(8)

        # Dynamically compute flatten size
        self.flatten_size = self._get_flatten_size()

        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, 2)

    def _get_flatten_size(self):
        """
        Pass a dummy input through conv and pool layers to calculate the flatten size.
        """
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, self.input_length)
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            flatten_size = x.view(1, -1).shape[1]
        return flatten_size

    def forward(self, x):
        x = self.ln(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return torch.sigmoid(x)
