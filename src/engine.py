import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.early_stopping import EarlyStopping

def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, 
                save_path="best_model.pth", eary_stopping_enabled=True,
                device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Trains a 1D CNN model and saves the trained model.

    Args:
        model (torch.nn.Module): The CNN model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs.
        lr (float): Learning rate.
        save_path (str): Path to save the trained model.
        device (str): Device to run the training ("cuda" or "cpu").

    Returns:
        model: Trained model.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    early_stopping = EarlyStopping(patience=20, enabled=eary_stopping_enabled, save_path="best_model.pth")

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()

            # assert labels.min() >= 0, f"Erro: Label negativa encontrada: {labels.min().item()}"
            # assert labels.max() < 3, f"Erro: Label fora do intervalo (max = {labels.max().item()}, esperado < {3})"

            optimizer.zero_grad()
            # inputs = inputs.squeeze(2)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}] -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if early_stopping(val_acc, model):
            print("⏹ Treinamento interrompido por Early Stopping!")
            break

    # Salvar o modelo treinado
    if not eary_stopping_enabled:    
        torch.save(model.state_dict(), save_path)
        print(f"Modelo salvo em {save_path}")

    return model



import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report


def test_model(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Tests a trained CNN model for bearing fault classification.

    Args:
        model (torch.nn.Module): The trained CNN model.
        test_loader (DataLoader): DataLoader for test data.
        device (str): Device to run the testing ("cuda" or "cpu").

    Returns:
        None (Prints accuracy and classification results).
    """

    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()

    test_loss = 0.0
    correct, total = 0, 0
    all_preds = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Predictions
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Store predictions and true labels for evaluation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = correct / total

    # Print evaluation metrics
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    return test_acc
