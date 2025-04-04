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
    dataset_uored = PtDataset("data/processed/uored", sample_size, transform=normalize)
    dataset_hust_4 = PtDataset("data/processed/hust_4", sample_size, transform=normalize)
    dataset_hust_5 = PtDataset("data/processed/hust_5", sample_size, transform=normalize)
    dataset_hust_6 = PtDataset("data/processed/hust_6", sample_size, transform=normalize)
    dataset_hust_7 = PtDataset("data/processed/hust_7", sample_size, transform=normalize)
    dataset_hust_8 = PtDataset("data/processed/hust_8", sample_size, transform=normalize)
    
    data_groups = dataset_cwru.group_by("extent_damage", sample_size=sample_size, 
                                   filter={"sampling_rate":"48000", "extent_damage":["007", "014", "021"]})
    groups, ids, y, start_position = data_groups

    n_splits=len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_splits) 
    
    # Aplicando GroupKFold
    total_accuracies = []
    print("-"*50)
    print("Datasets for pre-training: CWRU and Hust-all")
    print("Dataset for finetuning: CWRU")
    print("-"*50)
    for rep in range(repetition):
        print(f"Repetition {rep+1}")
        accuracies = []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(ids, y, groups)):
            early_stopper = EarlyStopping(patience=5, min_delta=0.005, verbose=False)
            early_stopper_ft = EarlyStopping(patience=5, min_delta=0.001, verbose=False)
            print(f"Fold {fold + 1}")        
            # print(ids[train_idx])
            group_info_train = (ids[train_idx], y[train_idx], start_position[train_idx])
            group_info_test = ( ids[test_idx], y[test_idx], start_position[test_idx])
            
            train_dataset_cwru = VibrationDataset(dataset_cwru, group_info_train, sample_size, transform=normalize)
            train_dataset = dataset_hust_4
                
            test_dataset = VibrationDataset(dataset_cwru, group_info_test, sample_size, transform=normalize)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32)

            model = CNN1D(num_classes=4)
            # torch.save(model.state_dict(), f"saved_models/cnn1d_f{fold}.pth")
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

            # training
            print("Starting training.")
            train(model, train_loader, criterion, optimizer, num_epochs=70, device=device, early_stopping=None)
            
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
