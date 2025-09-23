import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def validate_model(config):
    """
    Config esperado:
      - model: nn.Module
      - val_loader: DataLoader
      - criterion: BCEWithLogitsLoss (binário) ou CrossEntropyLoss (multiclasse)
      - device: 'cuda' ou 'cpu'
      - threshold (opcional, binário): default 0.5
      - metrics (opcional): dict nome -> func(y_true, y_pred)
    """
    model = config["model"]
    val_loader = config["val_loader"]
    criterion = config["criterion"]
    device = config.get("device", "cpu")
    threshold = float(config.get("threshold", 0.5))
    metrics = config.get("metrics", {})

    model.to(device).eval()
    val_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)  # binário: (B,) ou (B,1) | multiclasse: (B,C)

            # --- BINÁRIO: saída (B,) ou (B,1) ---
            if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.size(1) == 1):
                logits = outputs.view(-1)                 # (B,)
                loss = criterion(logits, y.float())       # BCEWithLogitsLoss pede alvo float
                probs = torch.sigmoid(logits)             # (B,)
                preds = (probs >= threshold).long()       # (B,)
                y_true.extend(y.long().cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

            # --- MULTICLASSE: saída (B,C) ---
            else:
                loss = criterion(outputs, y.long())       # CrossEntropyLoss
                preds = torch.argmax(outputs, dim=1)      # (B,)
                y_true.extend(y.long().cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

            val_loss += loss.item() * x.size(0)

    val_loss /= len(val_loader.dataset)

    results = {
        "val_loss": val_loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    for name, func in metrics.items():
        results[name] = func(y_true, y_pred)

    return results
