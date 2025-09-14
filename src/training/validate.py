# training/eval.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def validate_model(model, data_loader, device, criterion=None):
    """Evaluate a model and return metrics + concatenated logits/targets."""
    model.eval()
    total_loss = 0.0
    n = 0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            if criterion is not None:
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * y.size(0)
            n += y.size(0)

            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    # Metrics
    preds = logits_cat.argmax(dim=1).numpy() if logits_cat.ndim == 2 and logits_cat.size(1) > 1 \
            else (logits_cat.squeeze().numpy() > 0.5).astype(int)
    targets_np = targets_cat.numpy()

    acc = accuracy_score(targets_np, preds)
    f1  = f1_score(targets_np, preds, average="macro")

    metrics = {
        "loss": (total_loss / n) if (criterion is not None and n > 0) else None,
        "acc": float(acc),
        "f1": float(f1),
        "logits": logits_cat,     # tensors, para show_confusion_matrix
        "targets": targets_cat,
    }
    return metrics
