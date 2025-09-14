# logging
import sys
import logging
from src.utils import LoggerWriter
# ---
import os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, accuracy_score, balanced_accuracy_score
)

# Seus modelos
from src.models import MCNN_LSTM, MCNN_LSTM_LowFreq, TinyCNN1D

def fft_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Apply FFT to a 1D signal and return the normalized magnitude spectrum.
    Keeps only the positive half of the spectrum.
    """
    x_np = x.numpy()
    fft_vals = np.fft.fft(x_np)
    mag = np.abs(fft_vals)[:len(fft_vals)//2]   # half spectrum
    mag = mag / (np.linalg.norm(mag) + 1e-8)    # normalize
    return torch.from_numpy(mag.astype(np.float32))

# ============================================================
#                       DATASETS (FS=6k)
# ============================================================
class BearingDataset(Dataset):
    """
    Dataset multiclasse (B/I/N/O) para TREINO com janelas de 250 pontos @6kHz.
    """
    def __init__(self, root_dir, split="train"):
        self.root_dir = os.path.join(root_dir, split)
        self.class_map = {"B": 0, "I": 1, "N": 2, "O": 3}
        self.samples = []
        for fname in os.listdir(self.root_dir):
            if fname.endswith(".npy") and fname[0] in self.class_map:
                path = os.path.join(self.root_dir, fname)
                self.samples.append((path, self.class_map[fname[0]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path).astype(np.float32)  # 250 pts (treino)
        if x.ndim == 1:
            x = x[None, :]                    # (1, 250)
        return torch.from_numpy(x), label


class BearingDatasetBinary(Dataset):
    """
    Dataset binário (N vs F=B/I/O) para TREINO com janelas de 250 pts @6kHz.
    """
    def __init__(self, root_dir, split="train"):
        self.root_dir = os.path.join(root_dir, split)
        self.binary_map = {"N": 0, "F": 1}
        self.class_map  = {"N": 0, "F": 1}   # compatibilidade
        self.class_names = ["N", "F"]
        self.samples = []
        for fname in os.listdir(self.root_dir):
            if not fname.endswith(".npy"):
                continue
            c = fname[0]
            if c == "N":
                y = self.binary_map["N"]
            elif c in ("B", "I", "O"):
                y = self.binary_map["F"]
            else:
                continue
            self.samples.append((os.path.join(self.root_dir, fname), y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path).astype(np.float32)  # 250 pts
        if x.ndim == 1:
            x = x[None, :]                    # (1, 250)
        return torch.from_numpy(x), label


class FaultOnlyView(Dataset):
    """
    Visão somente-falha (B/I/O) a partir de BearingDataset (exclui N).
    Mapeia: B->0, I->1, O->2. Entrega (1,250); ajuste de comprimento é feito no forward.
    """
    fault_map = {"B": 0, "I": 1, "O": 2}

    def __init__(self, base_dataset: BearingDataset):
        self.samples = []
        for path, _y4 in base_dataset.samples:
            c = os.path.basename(path)[0]
            if c in self.fault_map:
                self.samples.append((path, self.fault_map[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y3 = self.samples[idx]
        x = np.load(path).astype(np.float32)  # 250 pts
        if x.ndim == 1:
            x = x[None, :]                    # (1, 250)
        return torch.from_numpy(x), y3


# ============================================================
#                    HELPERS / TRAIN / EVAL
# ============================================================
def pad_1d_lastdim_to(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Faz right-pad/crop na última dimensão para atingir target_len.
    Aceita x (C,L) ou (B,C,L).
    """
    if x.ndim == 2:
        L = x.shape[-1]
        if L == target_len:
            return x
        if L < target_len:
            pad = target_len - L
            return F.pad(x, (0, pad))
        else:
            return x[..., :target_len]
    elif x.ndim == 3:
        L = x.shape[-1]
        if L == target_len:
            return x
        if L < target_len:
            pad = target_len - L
            return F.pad(x, (0, pad))
        else:
            return x[..., :target_len]
    else:
        raise ValueError(f"Expected tensor with ndim 2 or 3, got {x.ndim}")

def compute_class_weights_from_dataset(ds, num_classes):
    labels = [lbl for _, lbl in ds.samples]
    cnt = Counter(labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        weights[c] = 1.0 / max(cnt.get(c, 0), 1)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

def safe_forward(model, xb, length_candidates=None):
    """
    Tenta forward com comprimentos (pad/crop) até alinhar ramos (evita 27 vs 28 etc.).
    Retorna (logits, L_usado).
    """
    if length_candidates is None:
        length_candidates = [256, 252, 260, 248, 264, 250, 244, 268]
    last_err = None
    for L in length_candidates:
        try:
            xbL = pad_1d_lastdim_to(xb, L)
            return model(xbL), L
        except RuntimeError as e:
            msg = str(e)
            if "Sizes of tensors must match" in msg or "size mismatch" in msg:
                last_err = e
                continue
            else:
                raise
    raise last_err if last_err is not None else RuntimeError("safe_forward: no candidate length worked.")

def train_one_epoch(model, loader, optimizer, device, ds_for_weights, num_classes, use_safe_forward=False, length_candidates=None):
    model.train()
    ce = nn.CrossEntropyLoss(weight=compute_class_weights_from_dataset(ds_for_weights, num_classes).to(device))
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        if use_safe_forward:
            logits, _ = safe_forward(model, xb, length_candidates)
        else:
            logits = model(xb)
        loss = ce(logits, yb)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss) * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, device, use_safe_forward=False, length_candidates=None):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if use_safe_forward:
            logits, _ = safe_forward(model, xb, length_candidates)
        else:
            logits = model(xb)
        loss = ce(logits, yb)
        loss_sum += float(loss) * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(yb.cpu().numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return loss_sum / total, correct / total, y_true, y_pred

# -------- 10×250 (sem preprocess) --------
def split_10x250(signal_2500: np.ndarray) -> torch.Tensor:
    """
    Recebe um vetor (2500,) e retorna tensor (10, 1, 250), sem preprocess.
    """
    assert signal_2500.ndim == 1 and signal_2500.shape[0] >= 2500
    segs = []
    for i in range(10):
        s = i * 250
        e = s + 250
        seg = signal_2500[s:e].astype(np.float32, copy=False)
        segs.append(seg[None, :])  # (1, 250)
    batch = np.stack(segs, axis=0)  # (10, 1, 250)
    return torch.from_numpy(batch)

@torch.no_grad()
def predict_soft_voting_segments(model, batch_10x: torch.Tensor, device, use_safe_forward=False, length_candidates=None):
    """
    Soft voting (média das probabilidades) sobre batch (10,1,L).
    Retorna (pred_idx, mean_probs ndarray, L_usado).
    """
    model.eval().to(device)
    if use_safe_forward:
        logits, L_used = safe_forward(model, batch_10x.to(device), length_candidates)
    else:
        logits = model(batch_10x.to(device))
        L_used = batch_10x.shape[-1]
    probs = F.softmax(logits, dim=-1)    # (10, C)
    mean_probs = probs.mean(dim=0).cpu().numpy()
    return int(mean_probs.argmax()), mean_probs, L_used


# ============================================================
#        AVALIAÇÃO DO GATE BINÁRIO NO REGIME 2500 (10×250)
# ============================================================
@torch.no_grad()
def eval_bin_soft_voting_on_folder(
    model_bin,
    root_dir,
    split="test",
    device="cuda",
    threshold=0.5,  # prob(F) >= threshold => F; senão N
    idx_F=1         # assumindo labels do binário: N=0, F=1
):
    """
    Avalia o classificador binário (N/F) no modo 2500->10×250 por soft voting.
    Imprime matriz e relatório N/F para comparação justa com o pipeline hierárquico.
    """
    folder = os.path.join(root_dir, split)
    y_true, y_pred = [], []

    for fname in os.listdir(folder):
        if not fname.endswith(".npy"):
            continue
        c = fname[0]
        if c not in ("B", "I", "N", "O"):
            continue

        true_bin = 0 if c == "N" else 1  # N=0, F=1
        raw = np.load(os.path.join(folder, fname)).astype(np.float32)
        if raw.ndim != 1 or raw.shape[0] < 2500:
            continue

        batch = split_10x250(raw)  # (10,1,250)
        _, mean_probs, _ = predict_soft_voting_segments(model_bin, batch, device, use_safe_forward=False)
        pF = float(mean_probs[idx_F])
        pred_bin = 1 if pF >= threshold else 0

        y_true.append(true_bin)
        y_pred.append(pred_bin)

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    labels_bin = [0, 1]  # N, F
    cm = confusion_matrix(y_true, y_pred, labels=labels_bin)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    print("\n===== BIN (2500→10×250) – GATE METRICS =====")
    print(f"Accuracy: {acc:.4f} | Balanced Acc: {bacc:.4f} | Threshold(F)={threshold:.2f}")
    print("Confusion Matrix N/F (rows=true, cols=pred):\n", cm)
    print("\nClassification Report (N/F):")
    print(classification_report(y_true, y_pred, labels=labels_bin, target_names=["N","F"], digits=4))
    return acc, bacc, cm


# ============================================================
#      AVALIAÇÃO HIERÁRQUICA (BIN N/F + FALHA B/I/O) – 2500
# ============================================================
@torch.no_grad()
def eval_hierarchical_soft_voting_on_folder(
    model_bin,           # MCNN_LSTM_LowFreq (N/F)
    model_fault,         # MCNN_LSTM (B/I/O)
    root_dir,
    split="test",
    device="cuda",
    bin_threshold=0.50,
    length_candidates_fault=None
):
    """
    Para cada arquivo .npy (2500 pts) em {root_dir}/{split}:
      1) Gate binário (N/F) por soft voting (10×250) com limiar (threshold).
      2) Se F -> multiclasse falha (B/I/O) por soft voting (10×250),
         usando safe_forward para alinhar ramos (pad/crop dinâmico).
      3) Métricas no espaço B/I/N/O.
    """
    class_map4 = {"B": 0, "I": 1, "N": 2, "O": 3}
    target_names4 = ["B", "I", "N", "O"]
    if length_candidates_fault is None:
        length_candidates_fault = [256, 252, 260, 248, 264, 250, 244, 268]

    folder = os.path.join(root_dir, split)
    y_true, y_pred = [], []

    for fname in os.listdir(folder):
        if not fname.endswith(".npy"):
            continue
        c = fname[0]
        if c not in class_map4:
            continue

        true_idx = class_map4[c]
        raw = np.load(os.path.join(folder, fname)).astype(np.float32)
        if raw.ndim != 1 or raw.shape[0] < 2500:
            continue

        batch = split_10x250(raw)  # (10,1,250)

        # 1) Gate binário com limiar
        _, mean_probs_bin, _ = predict_soft_voting_segments(model_bin, batch, device, use_safe_forward=False)
        pF = float(mean_probs_bin[1])  # F é índice 1 (N=0, F=1)
        is_failure = pF >= bin_threshold

        if not is_failure:
            final_idx4 = class_map4["N"]
        else:
            # 2) Falha B/I/O – com safe_forward (testa comprimentos)
            logits_fault, _ = safe_forward(model_fault, batch.to(device), length_candidates_fault)
            probs_fault = F.softmax(logits_fault, dim=-1)  # (10, 3)
            mean_probs_fault = probs_fault.mean(dim=0).cpu().numpy()
            fault_pred_idx = int(np.argmax(mean_probs_fault))
            final_idx4 = [class_map4["B"], class_map4["I"], class_map4["O"]][fault_pred_idx]

        y_true.append(true_idx)
        y_pred.append(final_idx4)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels4 = [0, 1, 2, 3]  # B, I, N, O nessa ordem

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1u = f1_score(y_true, y_pred, average="micro")
    cm = confusion_matrix(y_true, y_pred, labels=labels4)

    print("\n===== HIERARCHICAL TEST (10×250, soft voting, sem preprocess) =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-macro: {f1m:.4f}")
    print(f"F1-micro: {f1u:.4f}")
    print("Confusion Matrix (rows=true, cols=pred) [B,I,N,O]:\n", cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels4, target_names=target_names4, digits=4))
    return acc


# ============================================================
#                          MAIN
# ============================================================
def main(config):
    device = config["device"]
    os.makedirs(Path(config["ckpt_bin"]).parent, exist_ok=True)
    os.makedirs(Path(config["ckpt_fault"]).parent, exist_ok=True)

    # -------- 1) Treino Binário (N vs F) --------
    train_bin = BearingDatasetBinary(config["root"], split="train")
    test_bin  = BearingDatasetBinary(config["root"], split="test")

    ld_tr_bin = DataLoader(train_bin, batch_size=config["batch_size"], shuffle=True,
                           num_workers=config["num_workers"], pin_memory=True)
    ld_te_bin = DataLoader(test_bin,  batch_size=config["batch_size"], shuffle=False,
                           num_workers=config["num_workers"], pin_memory=True)

    model_bin = MCNN_LSTM_LowFreq(num_classes=2).to(device)
    opt_bin = torch.optim.Adam((p for p in model_bin.parameters() if p.requires_grad), lr=config["lr_bin"])

    best_bin = 0.0
    for ep in range(1, config["epochs_bin"] + 1):
        tr_loss, tr_acc = train_one_epoch(model_bin, ld_tr_bin, opt_bin, device, train_bin, num_classes=2, use_safe_forward=False)
        te_loss, te_acc, _, _ = eval_epoch(model_bin, ld_te_bin, device, use_safe_forward=False)
        print(f"[BIN] Epoch {ep:02d} | train: loss={tr_loss:.4f}, acc={tr_acc:.3f} | test: loss={te_loss:.4f}, acc={te_acc:.3f}")
        if te_acc > best_bin:
            best_bin = te_acc
            torch.save({"model": model_bin.state_dict()}, config["ckpt_bin"])
    print('Finish BINARY')
    # -------- 2) Treino Falha (B/I/O) --------
    train_4 = BearingDataset(config["root"], split="train")   # B/I/N/O
    test_4  = BearingDataset(config["root"], split="test")    # B/I/N/O

    train_fault = FaultOnlyView(train_4)  # apenas B/I/O
    test_fault  = FaultOnlyView(test_4)

    ld_tr_fault = DataLoader(train_fault, batch_size=config["batch_size"], shuffle=True,
                             num_workers=config["num_workers"], pin_memory=True)
    ld_te_fault = DataLoader(test_fault,  batch_size=config["batch_size"], shuffle=False,
                             num_workers=config["num_workers"], pin_memory=True)

    model_fault = MCNN_LSTM(num_classes=3).to(device)
    opt_fault = torch.optim.Adam((p for p in model_fault.parameters() if p.requires_grad), lr=config["lr"])

    length_candidates_fault = [256, 252, 260, 248, 264, 250, 244, 268]

    best_fault = 0.0
    for ep in range(1, config["epochs_fault"] + 1):
        tr_loss, tr_acc = train_one_epoch(
            model_fault, ld_tr_fault, opt_fault, device,
            train_fault, num_classes=3,
            use_safe_forward=True,
            length_candidates=length_candidates_fault
        )
        te_loss, te_acc, _, _ = eval_epoch(
            model_fault, ld_te_fault, device,
            use_safe_forward=True,
            length_candidates=length_candidates_fault
        )
        print(f"[BIO] Epoch {ep:02d} | train: loss={tr_loss:.4f}, acc={tr_acc:.3f} | test: loss={te_loss:.4f}, acc={te_acc:.3f}")
        if te_acc > best_fault:
            best_fault = te_acc
            torch.save({"model": model_fault.state_dict()}, config["ckpt_fault"])

    print(f"\nBest BIN acc: {best_bin:.4f}  | saved: {config['ckpt_bin']}")
    print(f"Best BIO acc: {best_fault:.4f} | saved: {config['ckpt_fault']}")

    # -------- 3) Avaliação do GATE BINÁRIO no regime 2500 (10×250) --------
    bin_threshold = config.get("bin_threshold", 0.50)
    eval_bin_soft_voting_on_folder(
        model_bin=model_bin,
        root_dir=config["root"],
        split="test",
        device=device,
        threshold=bin_threshold,
        idx_F=1
    )

    # -------- 4) TESTE HIERÁRQUICO (N/F → B/I/O) no regime 2500 --------
    real_acc = eval_hierarchical_soft_voting_on_folder(
        model_bin=model_bin,
        model_fault=model_fault,
        root_dir=config["root"],
        split="test",
        device=device,
        bin_threshold=bin_threshold,
        length_candidates_fault=length_candidates_fault
    )

    return real_acc, best_bin, best_fault


if __name__ == "__main__":
    experiment_key = "experiment"
    # sys.stdout = LoggerWriter(logging.info, experiment_key)
    def config(n):
        return {
            "root": f"data/processed/uored_lt_balanced/setup_{n}",  # contém train/ e test/ (treino: 250; teste: 2500)
            "batch_size": 64,
            "epochs_bin": 30,      # épocas do binário
            "epochs_fault": 10,    # épocas do modelo de falha
            "lr": 1e-4,
            "lr_bin": 1e-5,
            "num_workers": 2,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "ckpt_bin":   f"checkpoints/setup_{n}_bin.pt",
            "ckpt_fault": f"checkpoints/setup_{n}_bio.pt",
            # Limiar do gate binário: prob(F) >= threshold => F (tune 0.55~0.70 para reduzir FP N→F)
            "bin_threshold": 0.56,
        }
    total = []
    for _ in range(1):
        parcial = []
        for n in range(1, 11):
            print(f"\n--------------\nSetup {n}\n--------------")
            real_acc, best_bin, best_fault = main(config(n))
            parcial.append(real_acc)
        total.append(np.mean(parcial))
        print('*'*30)
        print(f'Parcial accuracy: {np.mean(parcial)}')
        print('*'*30)
    print('-'*30)
    print(f'\nTotal accuracy: {np.mean(total)}')
