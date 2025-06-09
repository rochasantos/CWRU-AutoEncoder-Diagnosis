import torch
from torch.utils.data import DataLoader, ConcatDataset
from src.training import train
from src.models import CNN7, CNN10
from src.data_processing import VibrationMapBuilder, VibrationDatasetFromMap


def pre_train(model, root_dir, lr, num_epochs, checkpoint_path, device='cuda'):   
    print(f"Model: {model}")
    
    if isinstance(root_dir, list):
        v_map = [VibrationMapBuilder(rd).get_split(val_ratio=0.2) for rd in root_dir]
        train_dataset = ConcatDataset([
            VibrationDatasetFromMap(vm[0], transform=None) for vm in v_map
        ])
        val_dataset = ConcatDataset([
            VibrationDatasetFromMap(vm[1], transform=None) for vm in v_map
        ])
    else:
        train_map, val_map = VibrationMapBuilder(root_dir).get_split(val_ratio=0.2)
        train_dataset = VibrationDatasetFromMap(train_map, transform=None)
        val_dataset = VibrationDatasetFromMap(val_map)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    _ = train(model, train_loader, val_loader, None, criterion, optimizer, 
                    num_epochs=num_epochs, device=device, checkpoint_path=checkpoint_path, 
                    checkpoint_mode="val_loss", use_class_weights=False)

    
if __name__ == "__main__":
    # Redirect console output to logger
    # experiment_title = f"uored_hust_cnn7_pretrain"
    # sys.stdout = LoggerWriter(logging.info, experiment_title)

    # Parameters
    lr = 1e-4
    num_epochs=20
    root_dir = "data/processed/paderborn_io"
    # root_dir = ["data/processed/hust_6205_io", "data/processed/uored_io"]
    checkpoint_path=f"checkpoint/cnn10_paderborn_io.pth"
    model = CNN10(input_length=4096)
    pre_train(model, root_dir, lr, num_epochs, checkpoint_path)
