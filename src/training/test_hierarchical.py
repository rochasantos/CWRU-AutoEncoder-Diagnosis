import torch
from tqdm import tqdm

def test_hierarchical(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Testing"):
            signals = signals.to(device)
            predictions = model(signals)  # retorna lista de strings: 'A', 'B', 'C', 'D'

            # Convert int labels to expected string labels
            label_str = []
            for label in labels:
                if label.item() == 0:
                    label_str.append('B')
                elif label.item() == 1:
                    label_str.append('I')
                elif label.item() == 2:
                    label_str.append('N')
                elif label.item() == 3:
                    label_str.append('O')

            for pred, true_label in zip(predictions, label_str):
                if pred == true_label:
                    correct += 1
                total += 1

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc
