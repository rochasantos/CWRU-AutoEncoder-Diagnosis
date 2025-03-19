import torch
from sklearn.metrics import accuracy_score, confusion_matrix

def test(model, test_loader, device):
    """
    Tests a CNN model and computes accuracy.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for test data.
        device (str): "cuda" for GPU or "cpu".

    Returns:
        float: Model accuracy.
        ndarray: Confusion matrix.
    """
    model.to(device)
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation to save memory
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)

            outputs = model(signals)
            _, preds = torch.max(outputs, 1)  # Get the class with the highest probability

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute accuracy and confusion matrix
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return accuracy, cm
