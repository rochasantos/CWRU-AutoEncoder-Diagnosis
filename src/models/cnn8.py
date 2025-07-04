import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN8(nn.Module):
    def __init__(self, input_length, num_classes=2):
        super(CNN8, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16)
        self.pool1 = nn.MaxPool1d(kernel_size=8)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16)
        self.pool2 = nn.MaxPool1d(kernel_size=8)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16)
        self.pool3 = nn.MaxPool1d(kernel_size=8)

        # Calcular o tamanho da saída após convoluções para o Linear
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            x = self.pool1(self.conv1(dummy_input))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            self._to_linear = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x