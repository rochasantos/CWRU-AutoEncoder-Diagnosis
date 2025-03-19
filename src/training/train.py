import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    Trains a CNN for vibration signal classification.

    Args:
        model (torch.nn.Module): CNN model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function (e.g., nn.CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam, SGD).
        num_epochs (int): Number of training epochs.
        device (str): "cuda" for GPU or "cpu".

    Returns:
        list: Training loss history.
        list: Training accuracy history.
    """
    model.to(device)
    model.train()  # Set model to training mode
    loss_history = []
    accuracy_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for signals, labels in progress_bar:
            signals, labels = signals.to(device), labels.to(device)

            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())

        # Compute average loss and accuracy for the epoch
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total

        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

    return loss_history, accuracy_history
