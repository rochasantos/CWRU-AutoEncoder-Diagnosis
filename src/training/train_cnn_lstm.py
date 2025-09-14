# train_mcnn_lstm.py
import os, math, time, json, random
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: sklearn just for f1 and confusion matrix (falls back gracefully)
try:
    from sklearn.metrics import f1_score, confusion_matrix
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# ---------- Utilities ----------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    max_grad_norm: Optional[float] = 1.0           # gradient clipping (LSTMs like this)
    label_smoothing: float = 0.0                   # optional smoothing for CE
    amp: bool = True                                # mixed precision
    scheduler: str = "cosine"                       # "cosine" | "step" | "none"
    step_size: int = 10
    gamma: float = 0.1
    patience: int = 7                                # early stopping patience (val loss)
    ckpt_dir: str = "checkpoints/mcnn_lstm"
    ckpt_name: str = "best.pt"
    seed: int = 42
    log_every: int = 50


class LabelSmoothingCE(nn.Module):
    """CrossEntropy with label smoothing; behaves like CE if smoothing=0."""
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        if self.smoothing <= 0.0:
            return F.cross_entropy(logits, target)
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


# ---------- Core train / eval ----------
def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0
) -> Dict[str, float]:
    """
    Single epoch training loop.
    - Keeps the data flow consistent with the diagram:
      x -> MCNN (two branches) -> fusion -> LSTM stack -> Dense -> logits.
    """
    model.train()
    loss_sum = 0.0
    correct = 0
    seen = 0

    t0 = time.time()
    for step, (x, y) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)     # expected shape (B, 1, 250)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if cfg.amp and torch.cuda.is_available():
            use_amp = cfg.amp and torch.cuda.is_available()
            autocast_ctx = torch.amp.autocast("cuda") if use_amp else torch.autocast(device_type="cpu", enabled=False)

            with autocast_ctx:
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if cfg.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

        bs = y.size(0)
        loss_sum += loss.item() * bs
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        seen += bs

        if cfg.log_every and step % cfg.log_every == 0:
            print(f"[train] epoch {epoch} step {step}/{len(loader)} "
                  f"loss {loss.item():.4f} acc {correct/seen:.3f}")

    return {
        "loss": loss_sum / max(1, seen),
        "acc":  correct / max(1, seen),
        "time": time.time() - t0
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    compute_cm: bool = True
) -> Dict[str, Any]:
    """Validation/test loop with optional F1 and confusion matrix."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0

    all_y: List[int] = []
    all_p: List[int] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        loss_sum += loss.item() * bs
        pred = logits.argmax(dim=1)

        all_y.extend(y.detach().cpu().tolist())
        all_p.extend(pred.detach().cpu().tolist())

        correct += (pred == y).sum().item()
        seen += bs

    acc = correct / max(1, seen)
    out = {"loss": loss_sum / max(1, seen), "acc": acc}

    if _HAS_SK:
        out["f1_macro"] = float(f1_score(all_y, all_p, average="macro"))
        if compute_cm:
            out["confusion_matrix"] = confusion_matrix(all_y, all_p)
    return out


def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if cfg.scheduler == "cosine":
        # Cosine with warmup (5% of total steps)
        total_steps = cfg.epochs * steps_per_epoch
        warmup = max(1, int(0.05 * total_steps))

        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            prog = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * prog))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown scheduler {cfg.scheduler}")


def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_classes: int,
    cfg: TrainConfig = TrainConfig(),
    class_weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Orchestrates the full training process according to the paper's pipeline.
    - Feature extractor + LSTMs are trained end-to-end with CE (optionally smoothed).
    - Uses AMP, gradient clipping, cosine/step scheduler, and early stopping on val loss.
    - Saves the BEST checkpoint by validation accuracy (tie-breaker: lower val loss).
    """
    set_seed(cfg.seed)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.ckpt_dir, cfg.ckpt_name)

    # Criterion
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = LabelSmoothingCE(cfg.label_smoothing)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    use_amp = cfg.amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Early stopping state
    best = {"val_acc": -1.0, "val_loss": float("inf")}
    patience_left = cfg.patience
    history = {"train": [], "val": []}

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg, scaler, epoch)
        if scheduler is not None:
            # Step per iteration for cosine warmup; here we emulate per-epoch step for StepLR
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()
        history["train"].append(tr)

        va = evaluate(model, val_loader, criterion, device, compute_cm=True)
        history["val"].append(va)

        # For cosine/warmup lambda scheduler that expects step per-iteration:
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            # step once per epoch using "last batch" notion
            scheduler.step()

        # Logging
        msg = (f"[epoch {epoch}/{cfg.epochs}] "
               f"train: loss={tr['loss']:.4f} acc={tr['acc']:.3f} | "
               f"val: loss={va['loss']:.4f} acc={va['acc']:.3f}")
        if "f1_macro" in va:
            msg += f" f1={va['f1_macro']:.3f}"
        print(msg)

        # Save best by accuracy (tie-breaker by lower loss)
        is_better = (va["acc"] > best["val_acc"]) or (math.isclose(va["acc"], best["val_acc"]) and va["loss"] < best["val_loss"])
        if is_better:
            best.update({"val_acc": va["acc"], "val_loss": va["loss"], "epoch": epoch})
            torch.save({"model_state": model.state_dict(),
                        "cfg": asdict(cfg),
                        "num_classes": num_classes}, ckpt_path)
            patience_left = cfg.patience
        else:
            patience_left -= 1

        # Early stopping (on no improvement in val acc/loss)
        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch}. Best @ {best['epoch']} acc={best['val_acc']:.3f}")
            break

    # Final validation report with confusion matrix (if sklearn present)
    final_val = history["val"][-1]
    if _HAS_SK and "confusion_matrix" in final_val:
        print("Confusion matrix (val):")
        print(final_val["confusion_matrix"])

    print(f"Best checkpoint: {ckpt_path}")
    return {"history": history, "best": best, "ckpt_path": ckpt_path}
