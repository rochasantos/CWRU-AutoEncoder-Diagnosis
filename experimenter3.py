import sys
import logging
from src.utils import LoggerWriter

import numpy as np

import pickle

import torch
from torch.utils.data import DataLoader, ConcatDataset
from src.training import train, test, EarlyStopping
from src.models import CNN1D, ArticleCNN1D
from src.data_processing import VibrationDataset, split_vibration_dataset, TransformDataAugmentation, VibrationMapBuilder, VibrationDatasetFromMap

def run_kfold(lr, num_epochs, history_path, num_repetitions, classes_key, device='cuda'):

    total_accuracy = []
    total_history = []

    root_dir = "data/processed"

    folds = [("fold1", "14, 21", "7"), ("fold2", "7, 21", "14"), ("fold3", "7, 14", "21")]
    builders = [VibrationMapBuilder(f"{root_dir}/{classes_key}/{fold[0]}/train") for fold in folds]
    for rep in range(num_repetitions):
        print(f"\n------- Repetition: {rep+1} / {num_repetitions} -------")
        accuracies = []
        history = []
        for i, fold in enumerate(folds): 

            print(f"\nFold {i+1}")
            print(f"Train datasets: {fold[1]}")
            print(f"Test dataset: {fold[2]}")   

            # transform = TransformDataAugmentation(prob=0.5, normalize=True)

            train_map, val_map = builders[i].get_split(val_ratio=0.10)

            train_dataset = VibrationDatasetFromMap(train_map, transform=None)
            val_dataset = VibrationDatasetFromMap(val_map)            
            test_dataset = VibrationDataset(data_dir=f"{root_dir}/{classes_key}/{fold[0]}/test")       
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)

            # Inicializa e treina o modelo
            
            model = CNN1D(input_length=9600, num_classes=2)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr) # weight_decay=1e-4
            criterion = torch.nn.CrossEntropyLoss()
            loss_history, accuracy_history, val_loss_history, val_accuracy_history = train(model, train_loader, val_loader, criterion, optimizer, 
                            num_epochs=num_epochs, device=device, early_stopping=None ) #EarlyStopping(patience=5, start_threshold=0.06, min_delta=0.005)

            # Avaliação final no conjunto de teste
            acc = test(model, test_loader, device=device, class_names=["B", "O"])
            
            history.append((loss_history, accuracy_history, val_loss_history, val_accuracy_history, acc))
            accuracies.append(acc)
        total_history.append(history)
        print(f"\nParcial Evaluation Summary")
        print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
        print(f"Standard Deviation: {np.std(accuracies):.4f}")
        total_accuracy.append(accuracies)
    print("\nFinal Evaluation Summary:")
    print("-------------------------------------------")
    print(f"Mean Accuracy by Fold: {np.round(np.mean(total_accuracy, axis=0), 4)}")
    print(f"Standard Deviation by Fold: {np.round(np.std(total_accuracy, axis=0), 4)}")
    print("-------------------------------------------")
    print(f"\nTotal Accuracy: {np.mean(np.mean(total_accuracy, axis=1)):.4f}")
    print(f"Standard Deviation: {np.std(total_accuracy):.4f}")
    print("-------------------------------------------")
    
    with open(history_path, "wb") as f:
        pickle.dump([total_history, total_accuracy], f)


if __name__ == "__main__":
    # Redirect console output to logger
    classes_key = "bi"
    experiment_title = f"cnn_{classes_key}"
    sys.stdout = LoggerWriter(logging.info, experiment_title)

    # Parameters
    num_repetitions = 10
    lr = 0.0001
    num_epochs=30
    history_path = f"pkl_files/{experiment_title}.pkl"

    run_kfold(lr, num_epochs, history_path, num_repetitions, classes_key)
