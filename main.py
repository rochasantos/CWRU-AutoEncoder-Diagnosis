# logging
import sys
import logging
from src.utils import LoggerWriter

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import MCNN_LSTM, MCNN_LSTM_LowFreq, BearingCNN1D, HierarchicalClassifier, ModelFactory
from src.data_processing import BearingDataset
from src.training import train_model, validate_model

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
#                          MAIN
# ============================================================
def main(config, bin_cfg, bio_cfg, val_cfg):
    device = config["device"]
    
    # Binary train
    bin_model, history, best_path = train_model(bin_cfg)
    print('Finish Binary Train')
    
    # Multiclass train
    bio_model, history, best_path = train_model(bio_cfg)
    print('Finish Multiclass Train')
    
    # Validation
    binary_clf = bin_model
    multi_clf = bio_model
    val_cfg['model'] = HierarchicalClassifier(binary_clf, multi_clf, device=device)
    val_results = validate_model(val_cfg)

    return val_results['accuracy'], val_results['confusion_matrix']

def make_loaders(root, batch_size=64, num_workers=2):
    tr_bin_ds = BearingDataset(f'{root}/bin', task="bin")
    val_bin_ds = BearingDataset(f'{root}/bin_val', task="bin")
    tr_bio_ds = BearingDataset(f'{root}/bio', task="bio")
    te_ds = BearingDataset(f'{root}/test', task="bino")

    tr_bin_loader = DataLoader(tr_bin_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True)
    val_bin_loader = DataLoader(val_bin_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    tr_bio_loader = DataLoader(tr_bio_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

    return tr_bin_ds, tr_bio_ds, te_ds, tr_bin_loader, tr_bio_loader, te_loader



if __name__ == "__main__":

    experiment_key = "experiment"
    sys.stdout = LoggerWriter(logging.info, experiment_key)
   
    # BINARY TRAIN CONFIG
    bin_model_fac = ModelFactory(model_class=MCNN_LSTM, num_classes=2)
    bin_cfg = {
        'model': bin_model_fac,
        'train_loader': None, 
        'val_loader': None,
        'epochs': 5,
        'optimizer': {'name': 'adam', 'lr': 1e-4, 'weight_decay': 1e-4},
        'scheduler': {'name': 'plateau', 'factor': 0.5, 'patience': 5, 'min_lr': 1e-6},
        'criterion': torch.nn.CrossEntropyLoss(),
        'amp': True,
        'grad_clip': 1.0,
        'early_stopping': {'patience': 30, 'min_delta': 0.0},
        'checkpoint_dir': 'checkpoints',
        'checkpoint_name': 'best.pt',
        'save_every_epoch': False,
        'log_interval': 50,
        'eval_interval': 1,
    }

    # MULTICLASS TRAIN CONFIG
    bio_model_fac = ModelFactory(model_class=MCNN_LSTM, num_classes=3)
    bio_cfg = {
        'model': bio_model_fac,
        'val_loader': None,
        'epochs': 10,
        'optimizer': {'name': 'adam', 'lr': 1e-4, 'weight_decay': 1e-4},
        'scheduler': {'name': 'plateau', 'factor': 0.5, 'patience': 5, 'min_lr': 1e-6},
        'criterion': torch.nn.CrossEntropyLoss(),
        'amp': True,
        'grad_clip': 1.0,
        'early_stopping': {'patience': 30, 'min_delta': 0.0},
        'checkpoint_dir': 'checkpoints',
        'checkpoint_name': 'best.pt',
        'save_every_epoch': False,
        'log_interval': 50,
        'eval_interval': 1,
    }

    # VAL CONFIG
    hier_clf = HierarchicalClassifier
    val_cfg = {
    "model": HierarchicalClassifier,
    "val_loader": None,
    "criterion": torch.nn.CrossEntropyLoss(),
    "device": "cuda",
}

    # EXPERIMENTER CONFIG
    exp_cfg = {
        "batch_size": 64,
        "lr_bin": 1e-5,
        "lr_bio": 1e-4,
        "num_workers": 2,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "bin_threshold": 0.56,
    }
    total = []
    for _ in range(1):
        parcial = []
        for n in range(1, 11):
            print(f"\n--------------\nSetup {n}\n--------------")
            
            root_dir = f"data/processed/uored_hierarchical/setup_{n}"            
            
            bin_ds, bio_ds, te_ds, bin_loader, bio_loader, te_loader = make_loaders(root_dir)
            bin_cfg['train_loader'] = bin_loader
            bio_cfg['train_loader'] = bio_loader
            val_cfg['val_loader'] = te_loader

            acc, conf_matrix = main(exp_cfg, bin_cfg, bio_cfg, val_cfg)
            print(f'Partial Accuracy: {acc}')
            print(conf_matrix)         
            parcial.append(acc)            
        total.append(np.mean(parcial))
        print('*'*30)
        print(f'Parcial accuracy: {np.mean(parcial)}')
        print('*'*30)
    print('-'*30)
    print(f'\nTotal accuracy: {np.mean(total)}')
