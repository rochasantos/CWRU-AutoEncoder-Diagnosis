import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .rae_classifier import RAEClassifier


def train_rae_classifier(rae, dataset, num_classes, num_epochs=30, batch_size=32, learning_rate=1e-3):
    """
    Treina o classificador baseado no encoder do RAE.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    latent_dim = 64 # rae.fc_mu.out_features  # Obtém o número correto da saída do encoder
    print(f"latent_dim: {latent_dim}")
    classifier = RAEClassifier(rae.encoder, latent_dim, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for x_batch, labels in dataloader:
            x_batch, labels = x_batch.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = classifier(x_batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}, Acc: {acc:.2f}%')
    
    torch.save(classifier.state_dict(), "rae_model.pth")

    print("✅ Classificador Treinado!")
