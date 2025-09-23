import torch
import torch.nn as nn
from torch.optim import Adam
from torch.amp import autocast, GradScaler

# ---------- helpers ----------

def _ensure_model(config):
    """Accept either an nn.Module instance or a callable/builder in config['model']."""
    model_cfg = config['model']
    if isinstance(model_cfg, nn.Module):
        return model_cfg
    if callable(model_cfg):  # e.g., lambda: MCNN_LSTM_Binary()
        return model_cfg()
    # Optional legacy: dict with 'build' method
    if hasattr(model_cfg, 'build') and callable(model_cfg.build):
        return model_cfg.build()
    raise ValueError("config['model'] must be an nn.Module, a callable that returns a model, or an object with .build().")

@torch.no_grad()
def _compute_pos_weight_from_loader(loader, device):
    """
    Compute pos_weight = N_neg / N_pos for BCEWithLogitsLoss.
    Assumes labels are {0,1}. Returns None if it can't compute.
    """
    if loader is None:
        return None
    pos = 0
    neg = 0
    for _, y in loader:
        # y can be (B,) or (B,1); convert to 1D long
        if y.dim() == 2 and y.size(-1) == 1:
            y = y.squeeze(-1)
        y = y.long()
        pos += int((y == 1).sum().item())
        neg += int((y == 0).sum().item())
    if pos == 0:
        return None
    return torch.tensor([neg / float(pos)], device=device)

@torch.no_grad()
def _binary_metrics_from_logits(logits, y_true, threshold=0.5):
    """
    logits: (N,) or (N,1) raw scores
    y_true: (N,) or (N,1) in {0,1}
    returns dict with acc, f1, precision, recall, confusion matrix (tn, fp, fn, tp)
    """
    if logits.dim() == 2 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    if y_true.dim() == 2 and y_true.size(-1) == 1:
        y_true = y_true.squeeze(-1)

    probs = torch.sigmoid(logits)
    y_pred = (probs >= threshold).long()

    y_true = y_true.long()

    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion": (tn, fp, fn, tp)
    }

# ---------- main training function ----------

def train_model(config):
    """
    Train a binary classifier with BCEWithLogitsLoss.
    Expected keys in `config` (minimal):
      - model: nn.Module OR callable returning model OR object with .build()
      - train_loader: PyTorch DataLoader
      - val_loader: (optional) DataLoader
      - epochs: int
      - optimizer: (optional) torch.optim.Optimizer instance; if absent, Adam is created with lr
      - lr: (optional) float (used if optimizer not provided)
      - device: (optional) 'cuda' or 'cpu' (auto if absent)
      - amp: (optional) bool to enable automatic mixed precision (default: False)
      - threshold: (optional) float for binarization (default: 0.5)
      - pos_weight: (optional) float or 1D tensor with shape [1]; if None, we try to compute from train_loader
    Returns:
      model, history, best_state (dict)
    """
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    amp_enabled = bool(config.get('amp', False))
    threshold = float(config.get('threshold', 0.5))

    model = _ensure_model(config).to(device)

    train_loader = config['train_loader']
    val_loader = config.get('val_loader', None)

    # Optimizer
    lr = float(config.get('lr', 1e-3))
    optimizer = Adam(model.parameters(), lr=lr)

    # pos_weight (optional for imbalance)
    pos_weight_cfg = config.get('pos_weight', None)
    if pos_weight_cfg is None:
        pos_weight = _compute_pos_weight_from_loader(train_loader, device)
    else:
        # Accept float or tensor
        if isinstance(pos_weight_cfg, (float, int)):
            pos_weight = torch.tensor([float(pos_weight_cfg)], device=device)
        elif torch.is_tensor(pos_weight_cfg):
            pos_weight = pos_weight_cfg.to(device).view(1)
        else:
            pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

    scaler = GradScaler("cuda", enabled=(amp_enabled and torch.cuda.is_available()))

    epochs = int(config['epochs'])
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_confusion": [],  # tuples (tn, fp, fn, tp)
    }

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # ---------- train ----------
        model.train()
        running_loss = 0.0
        n_train = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(x)                 # (B,) or (B,1)
                # ensure target dtype & shape
                yf = y
                if yf.dim() == 2 and yf.size(-1) == 1:
                    yf = yf.squeeze(-1)
                yf = yf.to(dtype=logits.dtype)    # BCEWithLogitsLoss requires float targets in [0,1]

                loss = criterion(logits, yf)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            running_loss += loss.detach().item() * bs
            n_train += bs

        train_loss = running_loss / max(1, n_train)
        history["train_loss"].append(train_loss)

        # ---------- validate ----------
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            n_val = 0

            all_logits = []
            all_targets = []

            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)

                    logits = model(x)
                    yf = y
                    if yf.dim() == 2 and yf.size(-1) == 1:
                        yf = yf.squeeze(-1)
                    yf = yf.to(dtype=logits.dtype)

                    loss = criterion(logits, yf)

                    bs = y.size(0)
                    val_running_loss += loss.item() * bs
                    n_val += bs

                    all_logits.append(logits.detach().float().view(-1))
                    all_targets.append(y.detach().long().view(-1))

            val_loss = val_running_loss / max(1, n_val)
            all_logits = torch.cat(all_logits, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            metrics = _binary_metrics_from_logits(all_logits, all_targets, threshold=threshold)

            history["val_loss"].append(val_loss)
            history["val_acc"].append(metrics["acc"])
            history["val_f1"].append(metrics["f1"])
            history["val_confusion"].append(metrics["confusion"])

            # keep best by F1 (usually safer on imbalance)
            if metrics["f1"] > best_val_f1:
                best_val_f1 = metrics["f1"]
                best_state = {
                    "epoch": epoch,
                    "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": metrics["acc"],
                    "val_f1": metrics["f1"],
                    "val_confusion": metrics["confusion"],
                    "pos_weight": (pos_weight.detach().cpu().numpy().tolist() if pos_weight is not None else None),
                }

            # Console print (compact)
            tn, fp, fn, tp = metrics["confusion"]
            print(f"Epoch {epoch:03d} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_acc={metrics['acc']:.4f} | "
                  f"val_f1={metrics['f1']:.4f} | "
                  f"CM=[tn:{tn} fp:{fp} fn:{fn} tp:{tp}]")
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

    return model, history, best_state
