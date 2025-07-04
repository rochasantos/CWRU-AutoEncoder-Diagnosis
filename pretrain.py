import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torchvision import transforms
from src.training import train

def pre_train(model, root_dir, lr, num_epochs, checkpoint_path, device='cuda'):   
    print(f"Model: {model}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),                 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])      

    train_dataset = ImageFolder("data/spectrograms/uored/train", transform=transform)
    val_dataset = ImageFolder("data/spectrograms/uored/val", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    _ = train(model, train_loader, val_loader, None, criterion, optimizer, 
                    num_epochs=num_epochs, device=device, checkpoint_path=checkpoint_path, 
                    checkpoint_mode="val_accuracy", use_class_weights=False)

    
if __name__ == "__main__":
    # Redirect console output to logger
    # experiment_title = f"uored_hust_cnn7_pretrain"
    # sys.stdout = LoggerWriter(logging.info, experiment_title)

    # Parameters
    lr = 1e-3
    num_epochs=20
    root_dir = None

    checkpoint_path=f"checkpoint/resnet18_uored_bi.pth"

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Sequential(
                                    torch.nn.Dropout(p=0.5),
                                    torch.nn.Linear(512, 2)
                                ) 

    pre_train(model, root_dir, lr, num_epochs, checkpoint_path)
