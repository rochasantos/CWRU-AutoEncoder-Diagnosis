import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from models.cnn1d import CNN1D
from src.dataset import VibrationDataset
from src.engine import train_model, test_model

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Experimenter function
def experimenter():
    
    tr_dataset = VibrationDataset("data/processed/tr_data_mix.h5")
    val_dataset = VibrationDataset("data/processed/val_data.h5")
    te_dataset = VibrationDataset("data/processed/te_data.h5")
    
    tr_dataloader = DataLoader(tr_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    te_dataloader = DataLoader(te_dataset, batch_size=32, shuffle=False)
    

    model = CNN1D(num_classes=4)
    trained_model = train_model(model, tr_dataloader, val_dataloader, num_epochs=10, lr=0.0001, device="cuda")

    test_model(trained_model, test_loader=te_dataloader)

if __name__ == "__main__":
    experimenter()
