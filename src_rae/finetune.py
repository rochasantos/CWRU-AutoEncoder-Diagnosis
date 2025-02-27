import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def finetune_rae_classifier(rae_classifier, dataset, num_epochs=20, batch_size=32, learning_rate=1e-4, freeze_encoder_layers=True, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Fine-tuning do RAEClassifier. Ajusta algumas camadas do encoder e treina o classificador.

    Args:
        rae_classifier: Modelo RAEClassifier previamente treinado.
        dataset: Dataset do PyTorch para fine-tuning.
        num_epochs: Número de épocas de treinamento.
        batch_size: Tamanho do batch.
        learning_rate: Taxa de aprendizado (default: 1e-4).
        freeze_encoder_layers: Se True, mantém algumas camadas do encoder congeladas.
        device: Dispositivo ("cuda" ou "cpu").
    """
    rae_classifier.to(device)
    rae_classifier.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 🔹 Definir quais partes do encoder serão ajustadas
    if freeze_encoder_layers:
        for param in rae_classifier.encoder.parameters():
            param.requires_grad = False  # Congela todas as camadas do encoder

        # 🔹 Apenas a última camada do encoder será ajustada
        for param in list(rae_classifier.encoder.parameters())[-2:]:
            param.requires_grad = True

    # 🔹 Otimizador apenas para os parâmetros treináveis
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, rae_classifier.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"🔍 Fine-tuning iniciado - Encoder {'parcialmente treinável' if freeze_encoder_layers else 'totalmente treinável'}")

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for x_batch, labels in dataloader:
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

        acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}, Acc: {acc:.2f}%')

    print("✅ Fine-tuning concluído!")
