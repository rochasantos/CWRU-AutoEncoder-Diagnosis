import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from src.training.early_stopping import EarlyStopping

def finetune_rae_classifier(rae_classifier, train_dataset, val_dataset, num_epochs=20, batch_size=32, learning_rate=1e-4, 
                            save_path="rae_cls.pth", save_best_model_path="best_model.pth", device="cuda", eary_stopping_enabled=False):
    
    print(f"Starting fine tuning...")
    print(f"Learning rate: {learning_rate}, number of epochs: {num_epochs}")

    rae_classifier.to(device)
    rae_classifier.train()

    # 🔹 Congela as camadas do encoder para evitar treino
    for param in rae_classifier.encoder.parameters():
        param.requires_grad = False  

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, rae_classifier.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=10, enabled=eary_stopping_enabled, save_path=save_best_model_path)

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for x_batch, labels in train_dataloader:
            x_batch, labels = x_batch.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = rae_classifier(x_batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        
        # 🔹 Validação
        rae_classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, labels in val_dataloader:
                x_batch, labels = x_batch.to(device), labels.to(device)
                logits = rae_classifier(x_batch)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_acc = 100 * correct / total
        rae_classifier.train()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_dataloader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

        if early_stopping(val_acc, rae_classifier):
            print("⏹ Treinamento interrompido por Early Stopping!")
            break

    # Salvar o modelo treinado
    torch.save(rae_classifier.state_dict(), save_path)
    print(f"Model saved in {save_path}")
    
    print("✅ Fine-Tuning Concluído!")
