import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import GroupKFold
from datasets import SU, CWRU, Hust, UORED
from datasets.vibration_dataset import VibrationDataset
from datasets.dataset import PtDataset
from src.models.cnn1d import CNN1D
from src.training.train import train
from src.training.test import test
from src.training.early_stopping import EarlyStopping

def normalize(signal):
    return (signal - signal.mean()) / (signal.std() + 1e-8)

def experimenter():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset_cwru = CWRU()
    sample_size = 6400
    num_epochs = 30
    repetition = 5

    # pre-training datasets
    dataset_cwru_007 = PtDataset("data/processed/cwru/007", sample_size, transform=normalize)
    dataset_cwru_014 = PtDataset("data/processed/cwru/014", sample_size, transform=normalize)
    dataset_cwru_021 = PtDataset("data/processed/cwru/021", sample_size, transform=normalize)
    dataset_uored = PtDataset("data/processed/uored", sample_size, transform=normalize)
    dataset_hust_4 = PtDataset("data/processed/hust_4", sample_size, transform=normalize)
    dataset_hust_5 = PtDataset("data/processed/hust_5", sample_size, transform=normalize)
    dataset_hust_6 = PtDataset("data/processed/hust_6", sample_size, transform=normalize)
    dataset_hust_7 = PtDataset("data/processed/hust_7", sample_size, transform=normalize)
    dataset_hust_8 = PtDataset("data/processed/hust_8", sample_size, transform=normalize)
    

    total_accuracies = []
    print("-"*50)
    print("Experimenter cross dataset: Hust 6204 -> CWRU")
    print("-"*50)
    for rep in range(repetition):
        print(f"Repetition {rep+1}")
        accuracies = []           
            
        train_loader = DataLoader(dataset_hust_5, batch_size=32, shuffle=True)
        test_loader = DataLoader(dataset_cwru_007, batch_size=32)

        model = CNN1D(num_classes=4)
        # torch.save(model.state_dict(), f"saved_models/cnn1d_f{fold}.pth")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # training
        print("Starting training.")
        train(model, train_loader, criterion, optimizer, num_epochs=60, device=device, early_stopping=None)
        
        # # finetuning
        # print("Starting finetuning.")
        # for param in model.parameters():
        #     param.requires_grad = False
        # for param in model.fc.parameters():
        #     param.requires_grad = True
        # criterion_ft = nn.CrossEntropyLoss()
        # optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
        # train(model, DataLoader(train_dataset_cwru, batch_size=32, shuffle=True), criterion_ft, optimizer_ft, 
        #       num_epochs=200, device=device, early_stopping=early_stopper_ft)
        
        # testing
        accuracy = test(model, test_loader, device)
        accuracies.append(accuracy)
        print("-" * 50)
        total_accuracies.append(accuracies)
    result_acc = [f"{acc*100:.2f}%" for acc in np.mean(total_accuracies, axis=0)]
    result_std = [f"{std*100:.2f}%" for std in np.std(total_accuracies, axis=0)]
    total_mean_acc = np.mean(total_accuracies)
    print(f"Mean Accuracies: {result_acc}")
    print(f"Standard Deviation: {result_std}")
    print(f"Total Mean Accuracy: {total_mean_acc*100:.2f}%")
