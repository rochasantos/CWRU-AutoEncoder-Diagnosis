import torch
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def test(model, test_loader, device="cuda", class_names=None):
    """
    Tests a CNN model and computes accuracy and confusion matrix.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for test data.
        device (str): "cuda" for GPU or "cpu".
        class_names (list or None): Optional list of class names for the confusion matrix.

    Returns:
        float: Test accuracy.
        ndarray: Confusion matrix.
    """
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)

            outputs = model(signals)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nTest Accuracy: {acc:.4f}")

    # Use provided class names or fallback to index numbers
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # Print formatted confusion matrix
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("\n[CONFUSION MATRIX]")
    print(df_cm)

    return acc
