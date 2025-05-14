import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_length, num_classes=4):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=16)
        # self.bn1 = nn.BatchNorm1d(num_features=32)
        self.pool1 = nn.MaxPool1d(kernel_size=8)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=16)
        # self.bn2 = nn.BatchNorm1d(num_features=32)
        self.pool2 = nn.MaxPool1d(kernel_size=8)

        # Cálculo dinâmico do tamanho da camada densa
        self._to_linear = self._get_conv_output(input_length)

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def _get_conv_output(self, input_length):
        x = torch.zeros(1, 1, input_length)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        return self.fc2(x)
