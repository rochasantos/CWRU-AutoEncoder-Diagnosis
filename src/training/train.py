import os
import math
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from collections import defaultdict
from time import time

def _accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

def _build_optimizer(model, opt_cfg):
    name = (opt_cfg.get('name') or 'adam').lower()
    lr = opt_cfg.get('lr', 1e-3)
    wd = opt_cfg.get('weight_decay', 0.0)
    if name == 'sgd':
        momentum = opt_cfg.get('momentum', 0.9)
        nesterov = opt_cfg.get('nesterov', True)
        return SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum, nesterov=nesterov)
    # default adam
    betas = opt_cfg.get('betas', (0.9, 0.999))
    eps = opt_cfg.get('eps', 1e-8)
    return Adam(model.parameters(), lr=lr, weight_decay=wd, betas=betas, eps=eps)

def _build_scheduler(optimizer, sch_cfg, train_loader, epochs, has_val):
    """Returns (scheduler, step_on) where step_on in {'epoch','iteration','val_metric','none'}"""
    if not sch_cfg or sch_cfg.get('name') is None:
        return None, 'none'
    name = sch_cfg['name'].lower()
    if name in ['plateau', 'reduceonplateau', 'reduce_lr_on_plateau']:
        # Step when validation accuracy plateaus (needs val)
        if not has_val:
            return None, 'none'
        factor = sch_cfg.get('factor', 0.5)
        patience = sch_cfg.get('patience', 5)
        cooldown = sch_cfg.get('cooldown', 0)
        min_lr = sch_cfg.get('min_lr', 1e-6)
        verbose = sch_cfg.get('verbose', True)
        return ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience,
                                 cooldown=cooldown, min_lr=min_lr, verbose=verbose), 'val_metric'
    if name in ['cosine', 'cosineannealinglr']:
        T_max = sch_cfg.get('t_max', epochs)
        eta_min = sch_cfg.get('eta_min', 0.0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min), 'epoch'
    if name in ['onecycle', 'onecyclelr']:
        # Needs steps_per_epoch
        max_lr = sch_cfg.get('max_lr', optimizer.param_groups[0]['lr'])
        steps_per_epoch = sch_cfg.get('steps_per_epoch', len(train_loader))
        pct_start = sch_cfg.get('pct_start', 0.3)
        div_factor = sch_cfg.get('div_factor', 25.0)
        final_div_factor = sch_cfg.get('final_div_factor', 1e4)
        anneal_strategy = sch_cfg.get('anneal_strategy', 'cos')
        return OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
                          pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor,
                          anneal_strategy=anneal_strategy), 'iteration'
    return None, 'none'

def _save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def train_model(config):
    """
    Expected keys in `config` (all optional except model, train_loader):
      - model: torch.nn.Module
      - train_loader: DataLoader
      - val_loader: DataLoader or None
      - device: 'cuda' or 'cpu' (default: auto)
      - epochs: int (default 20)
      - criterion: loss function (default CrossEntropyLoss)
      - optimizer: dict -> {name: 'adam'|'sgd', lr, weight_decay, ...}
      - scheduler: dict -> {name: 'plateau'|'cosine'|'onecycle', ...}
        * If 'plateau', it will monitor best validation accuracy.
      - amp: bool (default True)
      - grad_clip: float or None (default None)
      - early_stopping: dict or None -> {patience: int, min_delta: float}
      - checkpoint_dir: str (default 'checkpoints')
      - checkpoint_name: str (default 'best.pt')
      - save_every_epoch: bool (default False)
      - log_interval: int (default 50)
      - eval_interval: int (default 1)  # validate every N epochs
    Returns:
      history (dict) and best_ckpt_path (str or None)
    """
    model = config['model'].build()
    train_loader = config['train_loader']
    val_loader = config.get('val_loader')
    device = config.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = config.get('epochs', 20)
    criterion = config.get('criterion') or nn.CrossEntropyLoss()
    amp_enabled = config.get('amp', True)
    grad_clip = config.get('grad_clip', None)
    early_cfg = config.get('early_stopping') or {}
    early_patience = early_cfg.get('patience', None)
    early_min_delta = early_cfg.get('min_delta', 0.0)
    ckpt_dir = config.get('checkpoint_dir', 'checkpoints')
    ckpt_name = config.get('checkpoint_name', 'best.pt')
    save_every_epoch = config.get('save_every_epoch', False)
    log_interval = config.get('log_interval', 50)
    eval_interval = max(1, config.get('eval_interval', 1))

    model.to(device)
    optimizer = _build_optimizer(model, config.get('optimizer', {}))
    scheduler, sched_step_on = _build_scheduler(optimizer, config.get('scheduler', {}),
                                                train_loader, epochs, has_val=val_loader is not None)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and device == 'cuda'))

    history = defaultdict(list)
    best_val_acc = -float('inf')
    best_epoch = -1
    no_improve_epochs = 0
    best_ckpt_path = None
    start_time = time()

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        run_loss = 0.0
        run_acc = 0.0
        n_batches = 0

        for i, batch in enumerate(train_loader, 1):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                # Expect dict with 'inputs' and 'targets'
                x, y = batch['inputs'], batch['targets']

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled and device == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            acc = _accuracy(logits, y)
            run_loss += loss.item()
            run_acc += acc
            n_batches += 1

            if scheduler is not None and sched_step_on == 'iteration':
                scheduler.step()

            if log_interval and (i % log_interval == 0):
                print(f"Epoch {epoch:03d} | step {i:04d}/{len(train_loader)} | "
                      f"loss {loss.item():.4f} | acc {acc:.4f} | lr {optimizer.param_groups[0]['lr']:.6f}")

        tr_loss = run_loss / max(1, n_batches)
        tr_acc = run_acc / max(1, n_batches)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)

        # ---- Validate ----
        do_validate = (val_loader is not None) and (epoch % eval_interval == 0)
        if do_validate:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            vb = 0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        x, y = batch
                    else:
                        x, y = batch['inputs'], batch['targets']
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    if amp_enabled and device == 'cuda':
                        with torch.amp.autocast('cuda'):
                            logits = model(x)
                            loss = criterion(logits, y)
                    else:
                        logits = model(x)
                        loss = criterion(logits, y)

                    val_loss += loss.item()
                    val_acc += _accuracy(logits, y)
                    vb += 1
            val_loss /= max(1, vb)
            val_acc /= max(1, vb)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # ---- Scheduler on validation metric (best val acc) ----
            if scheduler is not None and sched_step_on == 'val_metric':
                scheduler.step(val_acc)

            # ---- Checkpoint on improvement ----
            improved = val_acc > (best_val_acc + early_min_delta)
            if improved:
                best_val_acc = val_acc
                best_epoch = epoch
                best_ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                _save_checkpoint({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scaler_state': scaler.state_dict() if amp_enabled and device == 'cuda' else None,
                    'best_val_acc': best_val_acc,
                    'config': config
                }, best_ckpt_path)
                no_improve_epochs = 0
                print(f"[Best] epoch {epoch} | val_acc {val_acc:.4f} | saved -> {best_ckpt_path}")
            else:
                no_improve_epochs += 1

            # ---- Early stopping (based on val acc) ----
            if early_patience is not None and no_improve_epochs >= early_patience:
                print(f"[EarlyStopping] No improvement for {no_improve_epochs} epochs "
                      f"(best epoch {best_epoch}, best val_acc {best_val_acc:.4f}).")
                break

            print(f"Epoch {epoch:03d} | train: loss {tr_loss:.4f}, acc {tr_acc:.4f} | "
                  f"val: loss {val_loss:.4f}, acc {val_acc:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.6f}")
        else:
            # No validation this epoch
            if scheduler is not None and sched_step_on == 'epoch':
                scheduler.step()
            print(f"Epoch {epoch:03d} | train: loss {tr_loss:.4f}, acc {tr_acc:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.6f}")

        # Optional: save every epoch (rolling checkpoints)
        if save_every_epoch:
            path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            _save_checkpoint({'epoch': epoch, 'model_state': model.state_dict()}, path)

    elapsed = time() - start_time
    print(f"Training finished in {elapsed/60:.1f} min. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}.")
    return model, dict(history), best_ckpt_path
