import torch
import torch.nn as nn
import torch.nn.functional as F

class PbUCNN(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.4, input_size=(3, 224, 224)):
        super(PbUCNN, self).__init__()

        # Convoluções principais (Conv → BN → ReLU → MaxPool)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (sem BN)
        self.conv4 = nn.Conv2d(128, 16, kernel_size=1, stride=1)  # redução de canais
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(16, 32, kernel_size=1, stride=1)  # expansão de canais

        # Flatten dinâmico
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            dummy_output = self._get_conv_output(dummy_input)
            self.flat_dim = dummy_output.view(1, -1).shape[1]

        # Fully connected
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.flat_dim, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def _get_conv_output(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x

    def forward(self, x):
        x = self._get_conv_output(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
