import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple CNN 1D model
# This model is a simple 1D CNN with two convolutional layers followed by max pooling 
# and two fully connected layers.
class CNN1D(nn.Module):
    def __init__(self, input_length, num_classes=2):
        super(CNN1D, self).__init__()

        # self.layer_norm = nn.LayerNorm(input_length)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=16)
        self.pool1 = nn.MaxPool1d(kernel_size=16)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=16)
        self.pool2 = nn.MaxPool1d(kernel_size=16)

        self._to_linear = self._get_conv_output(input_length)

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _get_conv_output(self, input_length):
        x = torch.zeros(1, 1, input_length)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x.view(1, -1).shape[1]

    def forward(self, x):
        # x = self.layer_norm(x)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
