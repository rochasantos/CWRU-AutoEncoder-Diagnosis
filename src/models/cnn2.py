import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2(nn.Module):
    def __init__(self, input_length, num_classes=2):
        super(CNN2, self).__init__()

        # Camada 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=16)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=8)

        # Camada 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=8)

        # Bottleneck
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1)  # compressão
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)  # transformação
        self.conv5 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1)  # expansão

        # Cálculo do tamanho da saída
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))
            self._to_linear = x.view(1, -1).shape[1]

        # Fully connected
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self._to_linear, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
