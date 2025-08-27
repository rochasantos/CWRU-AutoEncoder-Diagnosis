import sys
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from src.utils import LoggerWriter, show_confusion_matrix
from src.data_processing import VibrationDataset, BalancedBatchSampler
from src.training import train_model, validate_model
from src.models.model_factory import ModelFactory
from src.models import BearingCNN1D, CNNLSTMCompat, TinyCNN1D, BearingCNN1D_2Convs


def zscore(x):
    m = x.mean()
    s = x.std() + 1e-8
    return (x - m) / s


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders_fold(round_root, fold_name, batch_size, num_workers):
    """Build train/val loaders for a given fold and test loader from the round."""
    fold_root = os.path.join(round_root, fold_name)
    train_ds = VibrationDataset(os.path.join(fold_root, "train"))
    val_ds   = VibrationDataset(os.path.join(fold_root, "val"))
    test_ds  = VibrationDataset(os.path.join(round_root, "test"))

    applying_balanced_sampler = False
    if applying_balanced_sampler:
        train_sampler = BalancedBatchSampler(
            labels=train_ds.labels, batch_size=batch_size, shuffle=True, drop_last=True
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def infer_num_classes(ds):
    if hasattr(ds, "classes") and ds.classes:
        return len(ds.classes)
    if hasattr(ds, "targets") and ds.targets:
        return int(max(ds.targets)) + 1
    if isinstance(ds, ConcatDataset):
        union = set()
        for sub in ds.datasets:
            if hasattr(sub, "classes") and sub.classes:
                union.update(sub.classes)
        if union:
            return len(union)
        mx, found = -1, False
        for sub in ds.datasets:
            if hasattr(sub, "targets") and sub.targets:
                mx = max(mx, int(max(sub.targets)))
                found = True
        if found:
            return mx + 1
    try:
        ys = [int(ds[i][1]) for i in range(min(len(ds), 2048))]
        if ys:
            return max(ys) + 1
    except Exception:
        pass
    raise RuntimeError("Unable to infer num_classes from dataset.")


def _discover_round_paths(data_root):
    def is_round_dir(p):
        return os.path.isdir(os.path.join(p, "test")) and any(
            os.path.isdir(os.path.join(p, f"fold{i}")) for i in (1, 2, 3)
        )

    if is_round_dir(data_root):
        return [data_root]

    round_dirs = []
    if os.path.isdir(data_root):
        for name in sorted(os.listdir(data_root)):
            if name.startswith("round_"):
                candidate = os.path.join(data_root, name)
                if is_round_dir(candidate):
                    round_dirs.append(candidate)
    if not round_dirs:
        raise RuntimeError(f"Nenhum round válido encontrado em {data_root}")
    return round_dirs


def _build_model(config, num_classes, device):
    kwargs = dict(config.get("model_kwargs", {}))
    kwargs["num_classes"] = num_classes
    model = ModelFactory(
        model_class=config["model_class"],
        classifier_layer=None,
        **kwargs
    ).build().to(device)
    return model


def _train_and_test_fold(round_root, fold_name, config, device, label_names):
    """
    Train on (train/val) of the fold, then TEST on round's test set.
    Returns test_metrics (dict).
    """
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = make_loaders_fold(
        round_root, fold_name, config["batch_size"], config["num_workers"]
    )
    num_classes = infer_num_classes(train_ds)

    model = _build_model(config, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Train using provided training loop (with early/best checkpoint if it has)
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        config["epochs"],
    )

    # ---- TEST (only place where we show confusion matrix) ----
    test_metrics = validate_model(model=model, data_loader=test_loader, device=device, criterion=None)
    print(f"[{os.path.basename(round_root)} | {fold_name} | TEST] "
          f"Acc: {test_metrics['acc']:.4f} | F1(macro): {test_metrics['f1']:.4f}")
    show_confusion_matrix(test_metrics["logits"], test_metrics["targets"], label_names=label_names)

    return test_metrics


def run_repetitions(config):
    """
    For each round:
      - for each seed:
        - for each fold (fold1..fold3):
            train on (train/val) of the fold
            test on round's test
            print confusion matrix (TEST only)
      - summarize per round (mean across folds and seeds)
    Finally, summarize across rounds.
    """
    device = torch.device(
        config["device"] if config["device"] else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seeds = config["seeds"] if config["seeds"] else list(range(config["repeats"]))

    round_paths = _discover_round_paths(config["data_root"])

    all_round_acc = []
    all_round_f1  = []

    for round_path in round_paths:
        print("\n" + "="*70)
        print(f"🏁 Iniciando {os.path.basename(round_path)}")
        print("="*70)

        # Get label names from test set once (classes are the same)
        _, _, test_ds_tmp, _, _, _ = make_loaders_fold(
            round_path, "fold1", config["batch_size"], config["num_workers"]
        )
        label_names = getattr(test_ds_tmp, "classes", None)

        round_test_accs = []
        round_test_f1s  = []

        fold_names = [d for d in ("fold1", "fold2", "fold3")
                      if os.path.isdir(os.path.join(round_path, d))]
        if not fold_names:
            raise RuntimeError(f"Nenhuma fold encontrada em {round_path}")

        for i, seed in enumerate(seeds, 1):
            print(f"\n[Seed {i}/{len(seeds)}] seed={seed}")
            set_seed(seed)

            for fold_name in fold_names:
                metrics = _train_and_test_fold(round_path, fold_name, config, device, label_names)
                round_test_accs.append(metrics["acc"])
                round_test_f1s.append(metrics["f1"])

        # ---- Round summary (average over folds x seeds) ----
        r_acc_mean = float(np.mean(round_test_accs))
        r_acc_std  = float(np.std(round_test_accs, ddof=1)) if len(round_test_accs) > 1 else 0.0
        r_f1_mean  = float(np.mean(round_test_f1s))
        r_f1_std   = float(np.std(round_test_f1s, ddof=1)) if len(round_test_f1s) > 1 else 0.0

        print("\n----- Resumo do round -----")
        print(f"{os.path.basename(round_path)} | Acc: média={r_acc_mean:.4f} ± {r_acc_std:.4f} "
              f"| F1(macro): média={r_f1_mean:.4f} ± {r_f1_std:.4f}")

        all_round_acc.append(r_acc_mean)
        all_round_f1.append(r_f1_mean)

    # ---- Final summary across rounds ----
    acc_mean = float(np.mean(all_round_acc))
    acc_std  = float(np.std(all_round_acc, ddof=1)) if len(all_round_acc) > 1 else 0.0
    f1_mean  = float(np.mean(all_round_f1))
    f1_std   = float(np.std(all_round_f1, ddof=1)) if len(all_round_f1) > 1 else 0.0

    print("\n===== Sumarização Final (todos os rounds) =====")
    print(f"Rounds avaliados: {len(round_paths)}")
    print(f"Accuracy   : média={acc_mean:.4f} | desvio={acc_std:.4f}")
    print(f"Macro F1   : média={f1_mean:.4f} | desvio={f1_std:.4f}")


if __name__ == "__main__":
    config = {
        # Pode apontar para UM round (ex.: ".../cwru_2048_48k/round_4")
        # OU para a raiz que contém vários rounds (rodará todos).
        "data_root": "data/processed/cwru_2048_48k",
        "repeats": 1,
        "seeds": [0],
        "epochs": 15,
        "batch_size": 64,
        "lr": 8e-4,
        "num_workers": 4,
        "device": "",
        "model_class": BearingCNN1D,
        "model_kwargs": {"num_classes": 4}
    }
    run_repetitions(config)
