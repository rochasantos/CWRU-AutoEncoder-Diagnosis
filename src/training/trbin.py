import torch
from torch.utils.data import DataLoader

def train_binary(cfg):
    """
    Simple training loop for MCNN_LSTM_Binary with BearingNPYDataset.
    cfg dict must include:
        - model: nn.Module
        - train_dataset: Dataset
        - val_dataset: Dataset (can be None)
        - device: 'cuda' or 'cpu'
        - batch_size: int
        - epochs: int
        - lr: float
    """
    device = "cuda"
    model = cfg["model"].build().to("cuda")

    train_loader = cfg['train_loader']
    val_loader = None
    if cfg.get("val_dataset") is not None:
        val_loader = DataLoader(cfg["val_dataset"],
                                batch_size=cfg["batch_size"],
                                shuffle=False)

    # Binary classification → use BCEWithLogitsLoss
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)

            optimizer.zero_grad()
            logits = model(x)               # (B,)
            loss = criterion(logits, y)     # y must be float in BCE
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds.cpu() == y.long().cpu()).sum().item()
            total += x.size(0)

        train_acc = correct / total
        avg_loss = total_loss / total

        print(f"Epoch {epoch+1}/{cfg['epochs']} "
              f"| Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

        # Validation step
        if val_loader:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.float().to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    val_loss += loss.item() * x.size(0)
                    preds = (torch.sigmoid(logits) >= 0.5).long()
                    val_correct += (preds.cpu() == y.long().cpu()).sum().item()
                    val_total += x.size(0)

            val_acc = val_correct / val_total
            avg_val_loss = val_loss / val_total
            print(f"             | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if epoch == cfg["epochs"]-1:
            torch.save(model.state_dict(), "ckp_bin.pth")



    return model
