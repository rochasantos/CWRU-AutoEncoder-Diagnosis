import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN 1D model
# This model is a simple 1D CNN with three convolutional layers followed by max pooling
# and two fully connected layers.
# The model uses Batch Normalization after each convolutional layer
class CNN1D(nn.Module):
    def __init__(self, input_length, num_classes=2):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16)
        self.norm1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=8)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16)
        self.norm2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=8)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16)
        self.norm3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=16)

        # Correct dummy pass through all conv + norm + pool layers
        dummy_input = torch.zeros(1, 1, input_length)
        out = self.pool1(F.relu(self.norm1(self.conv1(dummy_input))))
        out = self.pool2(F.relu(self.norm2(self.conv2(out))))
        out = self.pool3(F.relu(self.norm3(self.conv3(out))))

        self._to_linear = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = self.pool3(F.relu(self.norm3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
