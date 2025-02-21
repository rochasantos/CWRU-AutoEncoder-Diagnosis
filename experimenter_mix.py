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
    repetitions = 5
    
    tr_dataset = ConcatDataset([
        VibrationDataset("data/processed/cwru_uored_da_2"),
        VibrationDataset("data/processed/cwru_hust"),
        VibrationDataset("data/processed/uored_hust"),
        # VibrationDataset("data/processed/cwru/007"),
        # VibrationDataset("data/processed/cwru/021"),
        # VibrationDataset("data/processed/uored"),
        # VibrationDataset("data/processed/hust"),
    ])
    fit_dataset = ConcatDataset([
        # VibrationDataset("data/processed/cwru_uored_da_2"),
        VibrationDataset("data/processed/cwru/007"),
        VibrationDataset("data/processed/cwru/021"),
        # VibrationDataset("data/processed/uored"),
    ])
    val_dataset = VibrationDataset("data/processed/cwru/014")
    te_dataset = VibrationDataset("data/processed/cwru/014")
    
    tr_dataloader = DataLoader(tr_dataset, batch_size=32, shuffle=True)
    fit_dataloader = DataLoader(fit_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    te_dataloader = DataLoader(te_dataset, batch_size=32, shuffle=False)
    
    accuracies = []
    for i in range(repetitions):
        model = CNN1D(num_classes=3)
        trained_model = train_model(model, tr_dataloader, val_dataloader, num_epochs=30, lr=0.0001, device="cuda")
        freeze_layers(trained_model, ["conv1", "conv2", "conv3", "conv4"])
        fit_model = train_model(trained_model, fit_dataloader, val_dataloader, num_epochs=80, lr=0.00001, device="cuda")
        accuracy = test_model(fit_model, test_loader=te_dataloader)
        accuracies.append(accuracy)
    accuracies = np.array(accuracies)
    acc_mean = np.mean(accuracies) * 100
    acc_std = np.std(accuracies) * 100
    print(f"Mean Accuracy: {np.round(acc_mean, 2)}%, Std: {np.round(acc_std, 2)}%")

if __name__ == "__main__":
    experimenter()
