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
from src.training import train_model, validate_model, train_model_mix, validate_model_majority_vote
from src.models.model_factory import ModelFactory
from src.models import BearingCNN1D, CNNLSTMCompat, TinyCNN1D, BearingCNN1D_2Convs


def zscore(x):
    # Works with np.ndarray or torch.Tensor
    m = x.mean()
    s = x.std() + 1e-8
    return (x - m) / s


# ----------------------------- Utilities -----------------------------
def set_seed(seed):
    """Set seeds for controlled stochasticity."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_directory_structure():
    """Ensure expected directories exist (idempotent)."""
    root_dir = "data/spectrograms"
    for severity in ["007", "014", "021"]:
        for label in ["N", "I", "O", "B"]:
            os.makedirs(os.path.join(root_dir, severity, label), exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)


def make_loaders(data_root, batch_size, num_workers):
    """Create train/val/test dataloaders from a fixed directory layout."""
    # train_ds = ConcatDataset([
    #     VibrationDataset(os.path.join(data_root, "train")),
    #     VibrationDataset(os.path.join("data/processed/cwru_da/round_1", "train"))  # Augmented data
    # ])
    train_ds = VibrationDataset(os.path.join(data_root, "train"))
    val_ds   = VibrationDataset(os.path.join(data_root, "val"))
    test_ds  = VibrationDataset(os.path.join(data_root, "test"))

    # Applying balanced batch sampler
    applying_balanced_sampler = False  # Set to False to use regular DataLoader
    if applying_balanced_sampler:
        print(f"Using BalancedBatchSampler for training with batch_size={batch_size}")
        train_sampler = BalancedBatchSampler(
            labels=train_ds.labels,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler, num_workers=num_workers
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
# ---------------------------------------------------------------------


def infer_num_classes(ds):
    """Infer number of classes from dataset or ConcatDataset in a robust way."""
    # Direct attributes (VibrationDataset exposes these)
    if hasattr(ds, "classes") and ds.classes:
        return len(ds.classes)
    if hasattr(ds, "targets") and ds.targets:
        return int(max(ds.targets)) + 1

    # ConcatDataset: union across sub-datasets
    if isinstance(ds, ConcatDataset):
        # Try classes first
        union_class_names = set()
        for sub in ds.datasets:
            if hasattr(sub, "classes") and sub.classes:
                union_class_names.update(sub.classes)
        if union_class_names:
            return len(union_class_names)

        # Fallback: use numeric targets across subs
        max_label = -1
        found = False
        for sub in ds.datasets:
            if hasattr(sub, "targets") and sub.targets:
                max_label = max(max_label, int(max(sub.targets)))
                found = True
        if found:
            return max_label + 1

    # Last resort: sample a few items
    try:
        ys = []
        n = min(len(ds), 2048)
        for i in range(n):
            _, y = ds[i]
            ys.append(int(y))
        if ys:
            return max(ys) + 1
    except Exception:
        pass

    raise RuntimeError("Unable to infer num_classes from dataset.")


def run_repetitions(config):
    """Repeat the experiment across seeds and summarize performance."""
    device = torch.device(
        config["device"] if config["device"] else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = make_loaders(
        config["data_root"], config["batch_size"], config["num_workers"]
    )
    num_classes = infer_num_classes(train_ds)

    if config["data_root_ft"]:
        train_ft_ds, val_ft_ds, test_ft_ds, train_ft_loader, val_ft_loader, test_ft_loader = make_loaders(
            config["data_root_ft"], config["batch_size"], config["num_workers"]
        )

    acc_list, f1_list = [], []
    seeds = config["seeds"] if config["seeds"] else list(range(config["repeats"]))

    for i, seed in enumerate(seeds, 1):
        print(f"\n[Rep {i}/{len(seeds)}] seed={seed}")
        set_seed(seed)

        # Build model kwargs with injected num_classes
        kwargs = dict(config.get("model_kwargs", {}))
        kwargs["num_classes"] = num_classes

        # Build model via ModelFactory
        model = ModelFactory(
            model_class=config["model_class"],
            classifier_layer=None,
            **kwargs
        ).build().to(device)

        # --- FIX: create criterion and optimizer, and pass them POSITIONALLY ---
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        # Pre-train
        model = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            config["epochs"],
        )

        # Fine-tune
        if config['data_root_ft']:
            print("Starting fine-tuning...")
            for p in model.parameters():
                p.requires_grad = False
            # for p in model.conv2.parameters():
            #     p.requires_grad = True
            for p in model.head.parameters():
                p.requires_grad = True


            model = train_model(
                model,
                train_ft_loader,
                val_ft_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(model.parameters(), lr=config["ft_lr"]),
                device=device,
                num_epochs=config["epochs_ft"]
            )


        # Validate on test set
        metrics = validate_model(model=model, data_loader=test_loader, device=device, criterion=None)
        acc, f1 = metrics["acc"], metrics["f1"]
        acc_list.append(acc)
        f1_list.append(f1)
        print(f"Test Acc: {acc:.4f} | Test F1 (macro): {f1:.4f}")
        # Show confusion matrix
        label_names = getattr(train_ds, "classes", None)  # usa nomes de pastas, se existir
        show_confusion_matrix(metrics["logits"], metrics["targets"], label_names=label_names)

    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list, ddof=1) if len(acc_list) > 1 else 0.0
    f1_mean = np.mean(f1_list)
    f1_std = np.std(f1_list, ddof=1) if len(f1_list) > 1 else 0.0

    print("\n===== Summary over repetitions =====")
    print(f"Repetitions: {len(seeds)}")
    print(f"Accuracy   : mean={acc_mean:.4f} | std={acc_std:.4f}")
    print(f"Macro F1   : mean={f1_mean:.4f} | std={f1_std:.4f}")


if __name__ == "__main__":
    # create_directory_structure()  # optional

    # Capture prints in your logging infra
    # experiment_key = "experiment"
    # sys.stdout = LoggerWriter(logging.info, experiment_key)

    config = {
        "data_root": "data/processed/cwru_2048_48k/round_4",  # Root directory for training data
        "data_root_ft": None, #"data/processed/cwru_12resampled/round_1", #"data/processed/cwru_12resampled/round_1",  # Fine-tuning data root
        "repeats": 1,
        "seeds": [0], #, 2, 3, 4, 5 or [] to use range(repeats)
        "epochs": 15,
        "epochs_ft": 20,  # Fine-tuning epochs
        "batch_size": 64,
        "lr": 8e-4,
        "ft_lr": 8e-4,
        "num_workers": 4,
        "device": "",
        "model_class": BearingCNN1D,
        "model_kwargs": {
            'num_classes':4
        }
    }
    run_repetitions(config)
