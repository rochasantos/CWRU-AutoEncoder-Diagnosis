import torch
import torch.nn as nn
import torch.nn.functional as F

class ADCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ADCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(10 * 2 * 2, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 10 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
