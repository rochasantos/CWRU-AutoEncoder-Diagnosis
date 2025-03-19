import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from datasets import SU, CWRU
from datasets.vibration_dataset import VibrationDataset
from src.models.cnn1d import CNN1D
from src.training.train import train
from src.training.test import test


def experimenter():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = CWRU()
    sample_size = 4200
    num_epochs = 30

    data_groups = dataset.group_by("hp", sample_size=sample_size, filter={"sampling_rate":"48000"})
    groups = np.array([t[0] for t in data_groups])  # Obtém os grupos (n_group)
    ids = np.array([t[1] for t in data_groups])  # Obtém os identificadores (id)
    y = np.array([t[2] for t in data_groups])  # Obtém os rótulos (label)
    start_position = np.array([t[3] for t in data_groups])


    # Criando GroupKFold com 5 divisões
    n_splits=len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_splits)

    # Aplicando GroupKFold
    for fold, (train_idx, test_idx) in enumerate(gkf.split(ids, y, groups)):
        print(f"Fold {fold + 1}")
        print(f"Number of train samples: {len(train_idx)}")
        print(f"Number of test samples: {len(test_idx)}")
        # print(ids[train_idx])
        group_info_train = (ids[train_idx], y[train_idx], start_position[train_idx])
        group_info_test = ( ids[test_idx], y[test_idx], start_position[test_idx])
        train_dataset = VibrationDataset(dataset, group_info_train, sample_size)
        test_dataset = VibrationDataset(dataset, group_info_test, sample_size)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        model = CNN1D(num_classes=4)
        torch.save(model.state_dict(), f"saved_models/cnn1d_f{fold}.pth")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        train(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device)
        test(model, test_loader, device)
        print("-" * 50)
