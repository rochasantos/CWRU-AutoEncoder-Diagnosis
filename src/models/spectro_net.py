import torch.nn as nn
import torchvision.models as models

class SpectroNet(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        
        z = self.encoder(x)     # (B, 3, 224, 224)
        print(z)
        return self.resnet(z)
