import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from src.training import train
import src.models as md
from src.data_processing import VibrationMapBuilder, VibrationDatasetFromMap


def pre_train(model_factory, dataset_dir, lr, num_epochs, device='cuda'):

    model = model_factory.build()
    print(f"Model: {model}")

    # uored
    train_map, val_map = VibrationMapBuilder(dataset_dir).get_split(val_ratio=0.2)
    train_dataset = VibrationDatasetFromMap(train_map, transform=None)
    val_dataset = VibrationDatasetFromMap(val_map, transform=None)

    # hust
    # train_map_h, val_map_h = VibrationMapBuilder("data/processed/hust/bi").get_split(val_ratio=0.2)
    # train_dataset_h = VibrationDatasetFromMap(train_map_h, transform=None)
    # val_dataset_h = VibrationDatasetFromMap(val_map_h)
    
    # train_loader = DataLoader(ConcatDataset([
    #     train_dataset, train_dataset_h]), batch_size=32, shuffle=True)
    # val_loader = DataLoader(ConcatDataset([val_dataset, val_dataset_h]), batch_size=32)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    checkpoint_path = f"checkpoint/cnn3conv_uored.pth"
    _, _, _, _ = train(model, train_loader, val_loader, criterion, optimizer, 
                    num_epochs=num_epochs, device=device, checkpoint_path=checkpoint_path)

    
if __name__ == "__main__":
    # Redirect console output to logger
    experiment_title = f"cnn_uored"
    # sys.stdout = LoggerWriter(logging.info, experiment_title)

    # Parameters
    lr = 0.0001
    num_epochs=20
    pre_train(lr, num_epochs)
