import sys
import logging
from src.utils import LoggerWriter

import numpy as np

import torch
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import StratifiedKFold, train_test_split

from src.training import train, test, EarlyStopping
from src.models import CNN1D
from src.data_processing import VibrationDataset


def run_kfold(dataset, k=3, batch_size=32, device='cuda'):
    # 1. Separa treino/val e teste fixo (80/20 estratificado)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        stratify=dataset.labels,
        random_state=42
    )

    print(f"Total: {len(dataset)} | Train/Val: {len(train_val_idx)} | Test: {len(test_idx)}")

    # 2. Extrai rótulos da parte treino/val para KFold
    train_val_labels = np.array(dataset.labels)[train_val_idx]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracies = []
    total_accuracy = []
    n_repetition=5
    for rep in range(n_repetition):
        print(f"Repetition: {rep+1} / {n_repetition}")
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_idx, train_val_labels)):
            print(f"\n🔁 Fold {fold+1}/{k}")

            # Mapear os índices para os índices globais do dataset original
            train_subset_idx = train_val_idx[train_idx]
            val_subset_idx = train_val_idx[val_idx]

            train_loader = DataLoader(Subset(dataset, train_subset_idx), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(Subset(dataset, val_subset_idx), batch_size=batch_size)
            test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size)

            # Inicializa e treina o modelo
            model = CNN1D(input_length=9600, num_classes=len(set(dataset.labels)))

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            criterion = torch.nn.CrossEntropyLoss()
            _,_,_,_ = train(model, train_loader, val_loader, criterion, optimizer, 
                            num_epochs=100, device=device, early_stopping=EarlyStopping(patience=5, start_threshold=0.04))

            # Avaliação final no conjunto de teste
            acc = test(model, test_loader, device=device, class_names=["B", "I"])
            accuracies.append(acc)
        
        mean_accuracy = np.mean(accuracies)
        total_accuracy.append(mean_accuracy)

        # Final evaluation summary
        print(f"Evaluation Summary for Fold {fold+1}.")
        print(f"Total Accuracy: {mean_accuracy:.4f}")
        print(f"Standard Deviation: {np.std(accuracies):.4f}")

    print("\nFinal Evaluation Summary:")
    print(f"Total Accuracy: {np.mean(total_accuracy):.4f}")
    print(f"Standard Deviation: {np.std(total_accuracy):.4f}")

if __name__ == "__main__":
    # Redirect console output to logger
    sys.stdout = LoggerWriter(logging.info, "log")

    data_dir = "data/processed/cwru_ib"
    run_kfold(
        dataset=VibrationDataset(data_dir),
        k=3
    )
