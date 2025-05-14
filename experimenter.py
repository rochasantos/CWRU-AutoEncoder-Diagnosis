import sys
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, accuracy_score
import logging
from src.utils import LoggerWriter
from src.training import train, test, EarlyStopping
from src.models import CNN3, SimpleCNN

# ----------------------------
# EarlyStopping
# ----------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

# ----------------------------
# # Generate train/validation DataLoaders
# ----------------------------
def get_dataloaders(data_dir, rate=0.2, batch_size=32, transform=None, shuffle=True):
    
    if isinstance(data_dir, list):
        datasets = [ImageFolder(data, transform=transform) for data in data_dir]
        dataset = ConcatDataset(datasets)
        classes = datasets[0].classes  # Assume all datasets share the same class structure
    else:
        dataset = ImageFolder(data_dir, transform=transform)
        classes = dataset.classes

    # Split dataset into train and validation subsets
    val_size = int(rate * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, classes

# ----------------------------
# Run the experiment
# ----------------------------
def main():
    # Redirect console output to logger
    sys.stdout = LoggerWriter(logging.info, "log")

    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define preprocessing pipeline
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize((129, 129)),
        transforms.ToTensor(),  # Keep RGB channels
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Define dataset directories
    tr_dir = "data/spectrograms/paderborn_artificial"
    ft_dir = "data/spectrograms/cwru_48k_7"
    te_dir = "data/spectrograms/cwru_48k_14"

    print(f"Train dataset: {tr_dir}")
    # print(f"Fine-tuning dataset: {ft_dir}")
    print(f"Test dataset: {te_dir}")

    accuracies = []

    n_repetitions = 2
    for rep in range(n_repetitions):
        print(f"\n\n====================\nRepetition {rep + 1}/{n_repetitions}\n====================")
        
        # Load training, fine-tunning, and test loaders
        tr_loader, tr_val_loader, _ = get_dataloaders(tr_dir, rate=0.2, transform=transform)
        ft_loader, ft_val_loader, _ = get_dataloaders(ft_dir, rate=0.1, transform=transform)
        te_loader, _, _ = get_dataloaders(te_dir, rate=0.0, transform=transform, shuffle=False)
        
        # Train the model
        model = SimpleCNN(num_classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion=torch.nn.CrossEntropyLoss()
        # _, _, _, _ = train( model=model,
        #                     train_loader=tr_loader,
        #                     val_loader=tr_val_loader,
        #                     criterion=criterion,
        #                     optimizer=optimizer,
        #                     num_epochs=30,
        #                     device=device,
        #                     scheduler=StepLR(optimizer, step_size=10, gamma=0.1),
        #                     early_stopping=None)
        
        # Save model after training
        # torch.save(model.state_dict(), "saved_models/cnn_pu.pth")
        # model.load_state_dict(torch.load("saved_models/cnn_pu.pth", weights_only=True))
        print("\n\n====================\nTest set evaluation training on PU\n====================")

        # Fine-tune the model
        model.load_state_dict(torch.load("saved_models/cnn_pu.pth", weights_only=True))
        # Freeze the initial layers
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the last layers
        for param in model.classifier.parameters():
            param.requires_grad = True
        # Parameters of the classifier will be trained
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        early_stopping = None # EarlyStopping(patience=5, min_delta=1e-4)
        # Train the model
        _, _, _, _ = train( model=model,
                            train_loader=ft_loader,
                            val_loader=ft_val_loader,
                            criterion=criterion,
                            optimizer=optimizer,
                            num_epochs=100,
                            device=device,
                            scheduler=scheduler,
                            early_stopping=early_stopping )
        
        # Save model after fine-tuning
        torch.save(model.state_dict(), "saved_models/cnn_finetuned.pth")

        # Evaluate the model using the custom test() function
        class_names = ["B", "I", "N", "O"]
        acc = test(model, te_loader, device, class_names=class_names)
        accuracies.append(acc)
    
    # Final evaluation summary
    print("\nFinal Evaluation Summary:")
    print(f"Total Accuracy: {np.mean(accuracies):.4f}")
    print(f"Standard Deviation: {np.std(accuracies):.4f}")    
        

if __name__ == "__main__":
    main()
