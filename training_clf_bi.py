import sys
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim

from src.utils import LoggerWriter
from src.training import train, test, th
from src.models import ModelFactory, BinaryCNN, HierarchicalClassifier
from src.data_processing import VibrationDataset



# K-fold cross-validation
def run_kfold(lr, num_epochs, history_path, num_repetitions, model_factory, labels_map, device='cuda'):
    print(f"Model Architecture: {model_factory.build()}")

    total_accuracy = []
    total_history = []

    root_dir = "data/processed/cwru"

    for rep in range(num_repetitions):
        print(f"\n------- Repetition: {rep + 1} / {num_repetitions} -------")
        accuracies = []
        history = []

        for n_fold in range(1, 4):  # 3 folds
            fold = f"fold{n_fold}"
            print(f"\n------- Fold: {n_fold} -------")

            # Dataset preparation            
            train_dataset = VibrationDataset(f"{root_dir}/{fold}/train", labels_map=labels_map, 
                                             selected_files=[])
            val_dataset = VibrationDataset(f"{root_dir}/{fold}/val", labels_map=labels_map)
            test_dataset = VibrationDataset(f"{root_dir}/{fold}/test", labels_map=labels_map)                         
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)
                       
            # Model initialization
            model = model_factory.build()           
            
            # Optionally load pre-trained weights
            # model.load_state_dict(torch.load("checkpoint/binary_uored.pth", weights_only=True)['model_state_dict'])
           
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            criterion = torch.nn.CrossEntropyLoss()
            
            checkpoint_path = f"checkpoint/clf_nf_f{n_fold}.pth"            

            # Training
            print("\nTreining a fault detector...")
            htr = train(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=num_epochs, device=device,
                checkpoint_path=checkpoint_path)
            
            print("\nCarregando melhores pesos...")
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True)['model_state_dict'])
            
            acc = test(model, test_loader, device=device, class_names=["N", "BIO"])
            
            history.append((htr, acc))
            accuracies.append(acc)

        total_history.append(history)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        print(f"\nPartial Evaluation Summary")
        print(f"Mean Accuracy: {mean_acc:.4f}")
        print(f"Standard Deviation: {std_acc:.4f}")

        total_accuracy.append(accuracies)

    # Final Summary
    mean_accuracy_by_fold = np.round(np.mean(total_accuracy, axis=0), 4)
    std_accuracy_by_fold = np.round(np.std(total_accuracy, axis=0), 4)

    print("\nFinal Evaluation Summary:")
    print("-------------------------------------------")
    print(f"Mean Accuracy by Fold: {mean_accuracy_by_fold}")
    print(f"Standard Deviation by Fold: {std_accuracy_by_fold}")
    print("-------------------------------------------")
    print(f"\nTotal Accuracy: {np.mean(mean_accuracy_by_fold):.4f}")
    print(f"Standard Deviation: {np.std(total_accuracy):.4f}")
    print("-------------------------------------------")

    # Save history
    with open(history_path, "wb") as f:
        pickle.dump([total_history, total_accuracy], f)


if __name__ == "__main__":
    # Setup Logger
    experiment_title = "lstm_005"
    root_dir = "data/processed/cwru"
    history_path = f"pkl_files/{experiment_title}.pkl"
    # sys.stdout = LoggerWriter(logging.info, experiment_title)

    # Parameters
    num_repetitions = 1
    lr = 0.0001
    num_epochs = 25
    labels_map = {'N': 0, 'I': 1, 'O': 1, 'B': 1}  # Fault detection labels

    model_factory = ModelFactory(model_class=BinaryCNN, input_channels=1, input_length=4096)

    run_kfold(lr, num_epochs, history_path, num_repetitions, model_factory, labels_map=labels_map)
