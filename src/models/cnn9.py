import torch
import torch.nn as nn
import torch.nn.functional as F

class ADCNN1D(nn.Module):
    def __init__(self, input_length, num_classes=4):
        super(ADCNN1D, self).__init__()

        # 1st Conv + MaxPool
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # 2nd Conv + MaxPool
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=10, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 3rd Conv + MaxPool
        self.conv3 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Calcular a saída do último layer conv+pool para conectar ao FC
        dummy_input = torch.zeros(1, 1, input_length)
        x = self._forward_features(dummy_input)
        self.flattened_size = x.view(1, -1).shape[1]

        # Fully connected
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _forward_features(self, x):
        x = self.pool1(F.relu(self.conv1(x)))   # -> [B, 5, ...]
        x = self.pool2(F.relu(self.conv2(x)))   # -> [B, 10, ...]
        x = self.pool3(F.relu(self.conv3(x)))   # -> [B, 10, ...]
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # ou softmax puro se preferir
