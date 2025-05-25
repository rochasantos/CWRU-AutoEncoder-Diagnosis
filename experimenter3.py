import sys
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim

from src.utils import LoggerWriter
from src.training import train, test
from src.models import ModelFactory, BinaryCNN, BinaryLSTM
from src.models.hpso_cnn_lstm import HPSO_CNN_LSTM
from src.data_processing import VibrationMapBuilder, VibrationDatasetFromMap

import torch
import numpy as np

def apply_fft(signal):
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)
    fft_magnitude = fft_magnitude[:len(fft_magnitude)//2]
    return fft_magnitude

def apply_fft_two_channels(signal):    
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)[:len(fft_result)//2]
    fft_phase = np.angle(fft_result)[:len(fft_result)//2]
    fft_magnitude = torch.tensor(fft_magnitude, dtype=torch.float32)
    fft_phase = torch.tensor(fft_phase, dtype=torch.float32)
    two_channels = torch.stack([fft_magnitude, fft_phase], dim=0)  # shape: (2, seq_len/2)
    return two_channels


# K-fold cross-validation
def run_kfold(lr, num_epochs, history_path, num_repetitions, model_factory, root_dir, device='cuda'):
    print(f"Model Architecture: {model_factory.build()}")

    total_accuracy = []
    total_history = []

    # Initialize builders for each fold
    builders = {
        f"builder_f{i+1}_train": VibrationMapBuilder(f"{root_dir}/fold{i+1}/train")
        for i in range(3)
    }
    builders.update({
        f"builder_f{i+1}_test": VibrationMapBuilder(f"{root_dir}/fold{i+1}/test")
        for i in range(3)
    })

    for rep in range(num_repetitions):
        print(f"\n------- Repetition: {rep + 1} / {num_repetitions} -------")
        accuracies = []
        history = []

        for n_fold in range(1, 4):  # 3 folds
            print(f"\n------- Fold: {n_fold} -------")

            # Dataset preparation
            train_map, val_map = builders[f"builder_f{n_fold}_train"].get_split(val_ratio=0.2)
            test_map = builders[f"builder_f{n_fold}_test"].get_test_map()

            train_dataset = VibrationDatasetFromMap(train_map, transform=None)
            val_dataset = VibrationDatasetFromMap(val_map, transform=None)
            test_dataset = VibrationDatasetFromMap(test_map, transform=None)   

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)

            # Model initialization
            model = model_factory.build()
            # Optionally load pre-trained weights
            # model.load_state_dict(torch.load("checkpoint/binary_uored.pth", weights_only=True)['model_state_dict'])

            # Example for freezing layers
            # for param in model.conv1.parameters():
            #     param.requires_grad = False
            # for param in model.conv2.parameters():
            #     param.requires_grad = False

            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            # criterion = torch.nn.CrossEntropyLoss()
            criterion = torch.nn.CrossEntropyLoss()
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = None # = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            checkpoint_path = f"checkpoint/best_model_fold{n_fold}.pth"

            # Training
            loss_history, accuracy_history, val_loss_history, val_accuracy_history = train(
                model, train_loader, val_loader, criterion, optimizer,
                num_epochs=num_epochs, device=device,
                checkpoint_path=checkpoint_path, early_stopping=None, scheduler=scheduler
            )

            # Testing
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True)['model_state_dict'])
            acc = test(model, test_loader, device=device, class_names=["B", "I"])

            history.append((loss_history, accuracy_history, val_loss_history, val_accuracy_history, acc))
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
    root_dir = "data/processed/cwru/bi"
    history_path = f"pkl_files/{experiment_title}.pkl"
    # sys.stdout = LoggerWriter(logging.info, experiment_title)

    # Parameters
    num_repetitions = 10
    lr = 0.0001
    num_epochs = 50

    model_factory = ModelFactory(model_class=BinaryCNN, input_channels=1, input_length=9600)

    run_kfold(lr, num_epochs, history_path, num_repetitions, model_factory, root_dir)
