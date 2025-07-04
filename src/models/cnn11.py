import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, in_channels, num_classes, input_length, num_filters1, num_filters2, kernel_size, dropout):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels=num_filters1, kernel_size=kernel_size, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=8)

        self.conv2 = nn.Conv1d(in_channels=num_filters1, out_channels=num_filters2, kernel_size=kernel_size, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=8)

        # Dummy input to determine layer norm shapes
        dummy_input = torch.zeros(1, in_channels, input_length)
        out1 = self.conv1(dummy_input)
        self.norm1 = nn.LayerNorm(out1.shape[1:])  # [C, L]

        out1 = self.pool1(F.relu(self.norm1(out1)))

        out2 = self.conv2(out1)
        self.norm2 = nn.LayerNorm(out2.shape[1:])  # [C, L]

        out2 = self.pool2(F.relu(self.norm2(out2)))

        self._to_linear = out2.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
