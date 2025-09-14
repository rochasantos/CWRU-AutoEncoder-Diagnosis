import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def validate_model(config):
    """
    Validate the model using the given configuration dictionary.
    Args:
        config (dict): Dictionary with parameters. Must include:
            - model: PyTorch model
            - val_loader: DataLoader for validation
            - criterion: Loss function
            - device: "cuda" or "cpu"
            - metrics (optional): dict with metric_name: function(y_true, y_pred)
    Returns:
        dict: validation results (loss and metrics)
    """
    model = config["model"]
    val_loader = config["val_loader"]
    criterion = config["criterion"]
    device = config.get("device", "cpu")
    metrics = config.get("metrics", {})

    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item() * x.size(0)

            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_loss /= len(val_loader.dataset)

    results = {"val_loss": val_loss}
    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    results["f1_micro"] = f1_score(y_true, y_pred, average="micro")
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    # Extra metrics if user passed
    for name, func in metrics.items():
        results[name] = func(y_true, y_pred)

    return results
