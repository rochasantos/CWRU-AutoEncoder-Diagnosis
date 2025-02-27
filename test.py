import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from models.cnn1d import CNN1D
from src.dataset import VibrationDataset
from src.engine import train_model, test_model

# Function to freeze layers
def freeze_layers(model, layers_to_freeze):
    for name, layer in model.named_children():
        if name in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Experimenter function
def experimenter():
    
    te_dataset = VibrationDataset("data/processed/cwru/014")
    
    te_dataloader = DataLoader(te_dataset, batch_size=32, shuffle=False)
    model = CNN1D(num_classes=3)
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    accuracy = test_model(model, test_loader=te_dataloader) * 100
    print(f"Accuracy: {np.round(accuracy, 2)}%")

if __name__ == "__main__":
    experimenter()
