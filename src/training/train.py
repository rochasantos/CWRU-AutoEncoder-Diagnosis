import torch
from collections import Counter

def train(model, train_loader, val_loader1=None, val_loader2=None,
          criterion=None, optimizer=None, num_epochs=10, device="cuda",
          checkpoint_path='best_model.pth', scheduler=None, early_stopping=None,
          use_class_weights=False, checkpoint_mode="val_accuracy"):

    model.to(device)
    loss_history = []
    accuracy_history = []
    val1_loss_history = []
    val1_accuracy_history = []
    val2_loss_history = []
    val2_accuracy_history = []

    best_acc = 0.0
    best_acc_acu = []
    best_loss = float('inf')

    # Class weights
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
    else:
        criterion = criterion or torch.nn.CrossEntropyLoss()

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

        model.eval()
        with torch.no_grad():

            # === Validação no val_loader1 ===
            if val_loader1:
                val1_loss = 0.0
                val1_correct = 0
                val1_total = 0
                for val_signals, val_labels in val_loader1:
                    val_signals, val_labels = val_signals.to(device), val_labels.to(device)
                    val_outputs = model(val_signals)
                    val1_loss += criterion(val_outputs, val_labels).item()
                    _, val_predicted = torch.max(val_outputs, 1)
                    val1_correct += (val_predicted == val_labels).sum().item()
                    val1_total += val_labels.size(0)
                avg_val1_loss = val1_loss / len(val_loader1)
                val1_acc = val1_correct / val1_total
                best_acc_acu.append(val1_acc)
                val1_loss_history.append(avg_val1_loss)
                val1_accuracy_history.append(val1_acc)

                print(f"Epoch [{epoch+1}/{num_epochs}] - "
                      f"Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f} - "
                      f"Val1 Loss: {avg_val1_loss:.4f}, Val1 Acc: {val1_acc:.4f}")

                save_checkpoint = False
                metric_value = val1_acc if checkpoint_mode == "val_accuracy" else avg_val1_loss
                if checkpoint_mode == "val_accuracy" and val1_acc > best_acc:
                    best_acc = val1_acc
                    save_checkpoint = True
                elif checkpoint_mode == "val_loss" and avg_val1_loss < best_loss:
                    best_loss = avg_val1_loss
                    save_checkpoint = True

                if save_checkpoint:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val1_loss,
                        'val_accuracy': val1_acc
                    }, checkpoint_path)
                    print(f"✅ Checkpoint salvo no epoch {epoch+1} com "
                          f"{'Val Accuracy' if checkpoint_mode == 'val_accuracy' else 'Val Loss'} {metric_value:.4f}")

                if early_stopping and early_stopping(avg_val1_loss, model):
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    break

            # === Validação no val_loader2 ===
            if val_loader2:
                val2_loss = 0.0
                val2_correct = 0
                val2_total = 0
                for val_signals, val_labels in val_loader2:
                    val_signals, val_labels = val_signals.to(device), val_labels.to(device)
                    val_outputs = model(val_signals)
                    val2_loss += criterion(val_outputs, val_labels).item()
                    _, val_predicted = torch.max(val_outputs, 1)
                    val2_correct += (val_predicted == val_labels).sum().item()
                    val2_total += val_labels.size(0)
                avg_val2_loss = val2_loss / len(val_loader2)
                val2_acc = val2_correct / val2_total
                val2_loss_history.append(avg_val2_loss)
                val2_accuracy_history.append(val2_acc)

                print(f"Epoch [{epoch+1}/{num_epochs}] - "
                      f"Val2 Loss: {avg_val2_loss:.4f}, Val2 Acc: {val2_acc:.4f}")

        # Caso nenhum val_loader esteja presente
        if not val_loader1 and not val_loader2:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
            if early_stopping and early_stopping(avg_loss, model):
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

        if scheduler:
            scheduler.step()

    return (loss_history, accuracy_history,
            val1_loss_history, val1_accuracy_history,
            val2_loss_history, val2_accuracy_history)
