import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.models.cnn1d import BearingCNN1D
from src.data_processing.bearing_dataset import BearingDataset
from src.training.training import train_model, evaluate_model

def zscore_normalize(x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True) + 1e-8
    return (x - mean) / std

# ---------------- Main Execution ----------------
def run_experimenter():
    # Parameters
    batch_size = 32
    learning_rate = 0.0001 #4.199e-5
    dropout = 0.5
    num_epochs = 40
    repetitions = 1
    classes = ['B', 'I']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc_rounds = []
    for round in ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9']:
        print(f"Round -> {round}")
        accuracies = []
        for rep in range(repetitions):
            print(f"Repetition: {rep}")
            # Load datasets            
            train_dataset = BearingDataset(
                root_dir=f"data/processed/cwru_bi/{round}", subset='train', classes=('B', 'I'),
                transform=zscore_normalize
            )
            val_dataset = BearingDataset(
                root_dir=f"data/processed/cwru_bi/{round}", subset='test', classes=('B', 'I'),
                transform=zscore_normalize
            )
            test_dataset = BearingDataset(
                root_dir=f"data/processed/cwru_bi/{round}", subset='test', classes=('B', 'I'),
                transform=zscore_normalize
            )           
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Model, loss, optimizer
            model = BearingCNN1D(in_channels=1, num_classes=1, p_drop=dropout) #(num_classes=len(classes), n_conv_layers=2, filters=32, kernel_size=32)
            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # criterion = nn.CrossEntropyLoss()
            # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Train and evaluate
            checkpoint_path = 'best_model.pth'
            train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, checkpoint_path=checkpoint_path)
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            # Avalie no conjunto de teste
            all_probs, all_preds, all_labels = evaluate_model(model, test_loader, device)

            # Calcule métricas
            acc = accuracy_score(all_labels, all_preds)
            print(f"Test Accuracy: {acc:.4f}")

            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, target_names=["Classe 0", "Classe 1"]))

            print("\nConfusion Matrix:")
            print(confusion_matrix(all_labels, all_preds))  
            # accuracy = evaluate_model(model, test_loader, device, classes)

            accuracies.append(acc)
        
        accuracies = np.array(accuracies)
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)        
        acc_rounds.append(mean_accuracy)

        print(f"\nModel Evaluation Summary:")
        print(f"Mean Accuracy: {mean_accuracy:.4f}")
        print(f"Standard Deviation: {std_accuracy:.4f}")
    
    total_acc = np.array(acc_rounds)
    total_acc_mean = np.mean(total_acc)

    print(total_acc_mean)