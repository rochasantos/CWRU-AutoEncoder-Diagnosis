# training/eval.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

def _is_multiclass(logits):
    # Multiclass if shape is [N, C] with C>1; otherwise treat as binary with sigmoid/threshold
    return logits.ndim == 2 and logits.size(1) > 1

def _majority_vote(preds_1d):
    # preds_1d: tensor [S] com predições inteiras por segmento
    counts = Counter(preds_1d.tolist())
    # Em caso de empate, escolhe o menor rótulo (comportamento determinístico e simples)
    return min(k for k, v in counts.items() if v == max(counts.values()))

def validate_model(model, data_loader, device, criterion=None,
                   eval_segment_len=None, eval_stride=None):
    """
    Evaluate a model and return metrics + concatenated logits/targets.

    If `eval_segment_len` is given and the incoming 1D signal length L > eval_segment_len,
    each sample is split into non-overlapping (or strided) segments of that length,
    the model runs on all segments, and the final prediction per original sample is
    decided by majority vote over the segment-level predictions.
    For the returned logits per sample, we aggregate by mean across its segments.

    Args:
        model: torch.nn.Module
        data_loader: yields (X, y). X shape expected [B, C, L] or [B, 1, L]
        device: torch.device
        criterion: loss function (optional)
        eval_segment_len: int or None. If None, behaves like the original function.
        eval_stride: int or None. If None, uses non-overlapping windows (stride = eval_segment_len).
    """
    model.eval()
    total_loss = 0.0
    n = 0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)  # [B, C, L]
            y = y.to(device)

            B = X.size(0)
            C = X.size(1) if X.ndim == 3 else 1
            L = X.size(-1)

            # Case A) No segmentation requested or not needed: run as usual
            if eval_segment_len is None or L <= (eval_segment_len or L):
                logits = model(X)
                if criterion is not None:
                    loss = criterion(logits, y)
                    total_loss += float(loss.item()) * y.size(0)
                n += y.size(0)
                all_logits.append(logits.detach().cpu())
                all_targets.append(y.detach().cpu())
                continue

            # Case B) Segment each sample in the batch along the time axis
            seg_len = int(eval_segment_len)
            stride = int(eval_stride) if eval_stride is not None else seg_len

            # Build a list with per-sample segment tensors; also track mapping back to sample idx
            seg_tensors = []
            seg_owner = []  # which original sample each segment belongs to
            for i in range(B):
                xi = X[i]  # [C, L]
                # Number of segments with the chosen stride
                if L < seg_len:
                    # Should not happen here, but guard anyway
                    start_idxs = [0]
                else:
                    start_idxs = list(range(0, max(L - seg_len + 1, 1), stride))
                # If last part is left out and you want to include it, you could append (L-seg_len)
                # For now we keep simple: non-overlapping or fixed stride windows from 0.

                for s in start_idxs:
                    seg = xi[..., s:s+seg_len]  # [C, seg_len]
                    if seg.size(-1) == seg_len:
                        seg_tensors.append(seg.unsqueeze(0))  # [1, C, seg_len]
                        seg_owner.append(i)

            if len(seg_tensors) == 0:
                # Fallback: run whole signal if no valid segments were created
                logits = model(X)
                if criterion is not None:
                    loss = criterion(logits, y)
                    total_loss += float(loss.item()) * y.size(0)
                n += y.size(0)
                all_logits.append(logits.detach().cpu())
                all_targets.append(y.detach().cpu())
                continue

            seg_batch = torch.cat(seg_tensors, dim=0).to(device)  # [S_total, C, seg_len]
            seg_logits = model(seg_batch)  # [S_total, num_classes] or [S_total, 1]

            # Compute per-segment predictions
            if _is_multiclass(seg_logits):
                seg_preds = seg_logits.argmax(dim=1)  # [S_total]
            else:
                # Binary: threshold at 0.5 after sigmoid
                seg_probs = torch.sigmoid(seg_logits.view(-1))
                seg_preds = (seg_probs > 0.5).long()

            # Aggregate back to per-sample: mean of logits for loss/logits; majority vote for pred
            # Prepare holders
            per_sample_logits = []
            per_sample_preds = []

            # To compute mean logits, we must gather logits per owner
            # For binary case, treat logits as single logit; keep shape consistent as [N, 1] or [N, C]
            num_classes = seg_logits.size(1) if _is_multiclass(seg_logits) else 1

            # Group indices by owner
            owners = [[] for _ in range(B)]
            for idx, o in enumerate(seg_owner):
                owners[o].append(idx)

            for i in range(B):
                idxs = owners[i]
                if len(idxs) == 0:
                    # If no segments for this sample (edge case), run full sample
                    logits_full = model(X[i:i+1])
                    per_sample_logits.append(logits_full)
                    if _is_multiclass(logits_full):
                        per_sample_preds.append(logits_full.argmax(dim=1).squeeze(0))
                    else:
                        p = torch.sigmoid(logits_full.view(-1))[0]
                        per_sample_preds.append((p > 0.5).long())
                    continue

                # Stack logits for this sample and take mean along segments
                li = seg_logits[idxs]  # [S_i, C] or [S_i, 1]
                li_mean = li.mean(dim=0, keepdim=True)  # [1, C] or [1, 1]
                per_sample_logits.append(li_mean)

                # Majority vote over segment-level preds for the final class
                pi = seg_preds[idxs]  # [S_i]
                mv = _majority_vote(pi.cpu())
                per_sample_preds.append(torch.tensor(mv, device=seg_logits.device))

            # Concatenate per-sample logits back to [B, C] or [B, 1]
            logits = torch.cat(per_sample_logits, dim=0)

            # Compute loss on aggregated logits
            if criterion is not None:
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * y.size(0)
            n += y.size(0)

            # Store outputs
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    # Concatenate across all batches
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    # Final predictions for metrics (based on aggregated logits)
    if _is_multiclass(logits_cat):
        preds = logits_cat.argmax(dim=1).numpy()
    else:
        preds = (torch.sigmoid(logits_cat.view(-1)) > 0.5).long().numpy()

    targets_np = targets_cat.numpy()
    acc = accuracy_score(targets_np, preds)
    f1 = f1_score(targets_np, preds, average="macro")

    metrics = {
        "loss": (total_loss / n) if (criterion is not None and n > 0) else None,
        "acc": float(acc),
        "f1": float(f1),
        "logits": logits_cat,   # aggregated per-sample logits
        "targets": targets_cat,
    }
    return metrics
