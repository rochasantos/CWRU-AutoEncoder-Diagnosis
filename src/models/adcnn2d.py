import torch
import torch.nn as nn
import torch.nn.functional as F

class ADCNN2D(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, input_size=(224, 224)):
        super(ADCNN2D, self).__init__()

        # 📦 Bloco 1: Conv -> BN -> ReLU -> Pool
        self.conv1 = nn.Conv2d(in_channels, 5, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # 📦 Bloco 2
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(10)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # 📦 Bloco 3
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(10)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # 🧠 Descobrir tamanho do flatten automaticamente
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *input_size)
            x = self._forward_features(dummy_input)
            self.flattened_size = x.view(1, -1).shape[1]

        # 🔗 Camadas totalmente conectadas
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _forward_features(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
