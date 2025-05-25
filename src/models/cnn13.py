import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_length, num_classes=2):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=16)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(8)
        self.norm1 = nn.LayerNorm(16)  # Corrigido: apenas canais

        self.conv2 = nn.Conv1d(16, 32, kernel_size=16)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(8)
        self.norm2 = nn.LayerNorm(32)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=16)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(16)
        self.norm3 = nn.LayerNorm(64)

        # Dummy input para calcular flatten
        dummy_input = torch.zeros(1, 1, input_length)
        out = self.pool1(F.relu(self.batch_norm1(self.conv1(dummy_input))))
        out = self.pool2(F.relu(self.batch_norm2(self.conv2(out))))
        out = self.pool3(F.relu(self.batch_norm3(self.conv3(out))))
        out = self._apply_layer_norm(out, self.norm3)

        self._to_linear = out.reshape(1, -1).shape[1]

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self._apply_layer_norm(x, self.norm1)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self._apply_layer_norm(x, self.norm2)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self._apply_layer_norm(x, self.norm3)

        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def _apply_layer_norm(self, x, norm):
        # x: [B, C, L]
        x = x.permute(0, 2, 1)  # -> [B, L, C]
        x = norm(x)
        x = x.permute(0, 2, 1)  # -> [B, C, L]
        return x
