import torch
from tqdm import tqdm

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device="cuda", scheduler=None, early_stopping=None):
    model.to(device)
    loss_history = []
    accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_signals, val_labels in val_loader:
                    val_signals, val_labels = val_signals.to(device), val_labels.to(device)
                    val_outputs = model(val_signals)
                    val_loss += criterion(val_outputs, val_labels).item()
                    _, val_predicted = torch.max(val_outputs, 1)
                    val_correct += (val_predicted == val_labels).sum().item()
                    val_total += val_labels.size(0)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            val_loss_history.append(avg_val_loss)
            val_accuracy_history.append(val_acc)

            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f} - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if early_stopping and early_stopping(avg_val_loss, model):
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
            if early_stopping and early_stopping(avg_loss, model):
                print(f"Early stopping at epoch {epoch+1}")
                break

        if scheduler:
            scheduler.step()

    return loss_history, accuracy_history, val_loss_history, val_accuracy_history
