import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

def test_classifier(classifier, test_dataset, class_names, batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Testa o classificador e exibe a acurácia e a matriz de confusão no console.

    Args:
        classifier: Modelo de classificação baseado no encoder do VAE.
        test_dataset: Dataset de teste do PyTorch (VibrationDataset).
        class_names: Lista com os nomes das classes (ex: ["Normal", "Falha I", "Falha O", "Falha B"]).
        batch_size: Tamanho do batch para avaliação.
        device: Dispositivo de execução ("cuda" ou "cpu").
    """
    classifier.eval()  # Coloca o modelo em modo de avaliação
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Não precisamos calcular gradientes durante o teste
        for x_batch, labels in test_loader:
            x_batch, labels = x_batch.to(device), labels.to(device)

            logits = classifier(x_batch)  # Faz as previsões
            predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)  # Obtém a classe predita

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # 🔹 Calcula a matriz de confusão
    cm = confusion_matrix(all_labels, all_preds)

    # 🔹 Exibe a acurácia
    accuracy = 100 * correct / total
    print(f"\n🎯 Acurácia no conjunto de teste: {accuracy:.2f}%\n")

    # 🔹 Exibe a matriz de confusão no console
    print("📊 Matriz de Confusão:")
    print(" " * 10, " ".join(f"{name[:6]:>6}" for name in class_names))  # Cabeçalho das classes
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>10} ", " ".join(f"{val:>6}" for val in row))  # Linhas da matriz

    return accuracy, cm
