import torch
from collections import Counter

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device="cuda",
          checkpoint_path='best_model.pth', scheduler=None, early_stopping=None,
          use_class_weights=False, checkpoint_mode="val_accuracy"):

    model.to(device)
    loss_history = []
    accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    best_acc = 0.0
    best_loss = float('inf')

    if use_class_weights:
        print("🔎 Calculando pesos de classe...")
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.tolist())

        counts = Counter(all_labels)
        num_samples = sum(counts.values())
        num_classes = len(counts)
        weights = [num_samples / (num_classes * counts[i]) for i in range(num_classes)]
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        print(f"✅ Pesos aplicados: {weights}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
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

        if val_loader:
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

            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f} - "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # === Condição para salvar o checkpoint ===
            save_checkpoint = False
            if checkpoint_mode == "val_accuracy" and val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint = True
            elif checkpoint_mode == "val_loss" and avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_checkpoint = True

            if save_checkpoint:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_acc
                }, checkpoint_path)

                metric_name = "Val Accuracy" if checkpoint_mode == "val_accuracy" else "Val Loss"
                metric_value = val_acc if checkpoint_mode == "val_accuracy" else avg_val_loss
                print(f"✅ Checkpoint salvo no epoch {epoch+1} com {metric_name} {metric_value:.4f}")

            if early_stopping and early_stopping(avg_val_loss, model):
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
            if early_stopping and early_stopping(avg_loss, model):
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

        if scheduler:
            scheduler.step()

    return loss_history, accuracy_history, val_loss_history, val_accuracy_history
