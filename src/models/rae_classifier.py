import torch
import torch.nn as nn


class RAEClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(RAEClassifier, self).__init__()
        self.encoder = encoder

        for param in self.encoder.parameters():
            param.requires_grad = False  

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),  
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        with torch.no_grad():
            encoded = self.encoder(x).view(batch_size, -1)  # Flatten para MLP            
        logits = self.classifier(encoded)
        return logits
