import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def pretrain(model, train_loader, num_classes=4, device="cuda", epochs=25, lr=1e-4,
             adjust_head=True, print_every=100, optimizer=None, criterion=None):
    """
    Pre-train a ResNet model on PNG spectrograms organized as:
      data_dir/
        B/*.png
        I/*.png
        N/*.png
        O/*.png

    Trains on the entire directory (no validation) and returns the updated model.

    Args:
      model: a torchvision ResNet model (e.g., resnet18) already instantiated.
      data_dir: path to 'pre_train' directory containing class subfolders (B, I, N, O).
      device: 'cuda' or 'cpu'.
      epochs: number of epochs for pre-training.
      lr: learning rate for the optimizer.
      batch_size: mini-batch size.
      num_workers: dataloader workers.
      adjust_head: if True, adjusts model.fc to match the number of classes in data_dir.
      print_every: print batch-level progress every N steps (0 to disable).
      optimizer: optional custom optimizer; if None, uses Adam over all parameters.
      criterion: optional loss; if None, uses CrossEntropyLoss.

    Returns:
      The pre-trained model (same instance).
    """
  
    # Optionally adjust the classifier head to match the dataset classes
    if adjust_head and hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        if model.fc.out_features != num_classes:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

    model.to(device)    

    # Default loss and optimizer if not provided
    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    global_step = 0

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (imgs, targets) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            # Accumulate simple metrics
            running_loss += loss.item() * imgs.size(0)
            _, preds = logits.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            global_step += 1
            if print_every and (i % print_every == 0):
                print(f"[pretrain] epoch {epoch:03d}/{epochs} | step {i:05d} | loss={loss.item():.4f}")

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        print(f"Epoch {epoch:03d}/{epochs} | pretrain: loss={epoch_loss:.4f}, acc={epoch_acc:.4f} (samples={total})")

    return model
