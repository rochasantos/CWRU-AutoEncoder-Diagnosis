import torch
import torch.nn as nn

class CNN3(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN3, self).__init__()

        # Initial convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 3x3
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 5x5 receptive field
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 7x7 receptive field
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),       
            nn.BatchNorm2d(64)
        )

        # Bottleneck block
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1, stride=1),
            nn.ReLU()
        )

        # Adaptive pooling to flatten safely
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (B, 32, 1, 1)

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, num_classes), 
            # nn.Softmax(dim=1) # CrossEntropyLoss or Softmax
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.bottleneck(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x