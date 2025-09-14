# src/training/train.py
import os
import time
import torch
from torch import amp

def _get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg.get("lr", None)

def _safe_macro_f1(y_true, y_pred):
    try:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        return 0.0
        

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs,
                mixed_precision=True, ckpt_path="checkpoints/best_val_acc.pt"):
    """
    Minimal training loop with per-epoch logs.
    Behavior:
      - With val_loader: track best-by-val-acc and save checkpoint; restore best at the end.
      - Without val_loader: do NOT use best-checkpointing; save final model only (last epoch).
    """
    model.to(device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = amp.GradScaler(device_type, enabled=mixed_precision)

    use_validation = val_loader is not None

    # Best checkpoint tracking only if we have validation
    best_acc = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0

        for x, y in train_loader:
            x = x.to(device).float()
            y = y.to(device).long()

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type, enabled=mixed_precision):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss_sum += loss.item() * x.size(0)
            tr_correct  += (logits.detach().argmax(1) == y).sum().item()
            tr_total    += y.numel()

        tr_loss = tr_loss_sum / max(tr_total, 1)
        tr_acc  = tr_correct  / max(tr_total, 1)

        # ---- optional validation ----
        val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
        if use_validation:
            model.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0
            y_true_list, y_pred_list = [], []

            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device).float()
                    y = y.to(device).long()
                    with amp.autocast(device_type, enabled=mixed_precision):
                        logits = model(x)
                        loss = criterion(logits, y)
                    val_loss_sum += loss.item() * x.size(0)
                    preds = logits.argmax(1)
                    val_correct += (preds == y).sum().item()
                    val_total   += y.numel()
                    y_true_list.append(y.cpu())
                    y_pred_list.append(preds.cpu())

            if val_total > 0:
                val_loss = val_loss_sum / val_total
                val_acc  = val_correct / val_total
                y_true   = torch.cat(y_true_list).numpy()
                y_pred   = torch.cat(y_pred_list).numpy()
                val_f1   = _safe_macro_f1(y_true, y_pred)

        epoch_time = time.time() - t0
        lr = _get_lr(optimizer)

        # Logging: print val metrics only if validation is used
        if use_validation:
            print(
                f"Epoch {epoch:03d}/{num_epochs} | "
                f"lr={lr:.6f} | "
                f"train: loss={tr_loss:.4f}, acc={tr_acc:.4f} | "
                f"val: loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f} | "
                f"time={epoch_time:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{num_epochs} | "
                f"lr={lr:.6f} | "
                f"train: loss={tr_loss:.4f}, acc={tr_acc:.4f} | "
                f"time={epoch_time:.1f}s"
            )

        # ---- checkpointing ----
        if use_validation:
            # Best-by-validation-accuracy checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                if ckpt_path:
                    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
                    torch.save({"model_state": best_state, "epoch": epoch, "val_acc": float(best_acc)}, ckpt_path)
                    print(f"  -> New best val_acc={best_acc:.4f}. Checkpoint saved to: {ckpt_path}")

    # ---- finalize ----
    if use_validation:
        # Restore best weights (if any) before returning
        if best_state is not None:
            model.load_state_dict(best_state)
        elif ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
    else:
        # No validation: save final model (last epoch) only
        if ckpt_path:
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save({"model_state": model.state_dict(), "epoch": num_epochs}, ckpt_path)
            print(f"Final model (last epoch) saved to: {ckpt_path}")

    return model
