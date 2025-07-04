import sys
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torchvision import transforms

from src.utils import LoggerWriter
from src.training import train, test
from src.models import CNN7
from src.models.model_factory import ModelFactory
from src.data_processing import VibrationDataset

# learning rate scheduler
def get_scheduler(optimizer, num_epochs=10, warmup_epochs=5):
    """
    Returns a learning rate scheduler that linearly increases the learning rate
    for the first `warmup_epochs` epochs and then decays it exponentially.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.90 ** (epoch - warmup_epochs)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# K-fold cross-validation
def run_kfold(root_dir, lr, num_epochs, num_repetitions, model_factory, 
              labels_map, use_class_weights=False, checkpoint_pretrain=None, device='cuda'):
    print(f"Model Architecture: {model_factory.build()}")

    classes_key = "".join(labels_map.keys()).lower()

    total_accuracy = []
    total_history = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet18 expected input
        transforms.ToTensor(),          # Convert to Tensor
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(15),
        transforms.Normalize(           # Normalize using ImageNet statistics
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    for rep in range(num_repetitions):
        print(f"\n------- Repetition: {rep + 1} / {num_repetitions} -------")
        accuracies = []
        history = []

        for n_fold in range(1, 4):  # 3 folds
            fold = f"fold{n_fold}"
            print(f"\n------- Fold: {n_fold} -------")
          
            train_dataset = ImageFolder(f"{root_dir}/{fold}/train", transform=transform)
            val_dataset = ImageFolder(f"{root_dir}/{fold}/val", transform=transform)
            test_dataset = ImageFolder(f"{root_dir}/{fold}/test", transform=transform)                         
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)
                       
            # Model initialization
            model = model_factory.build()

            # for name, param in model.named_parameters():
            #     if 'layer4' not in name and 'fc' not in name:
            #         param.requires_grad = False       
            
            if checkpoint_pretrain:
                print("Starting fine-tuning.")
                # Optionally load pre-trained weights
                model.load_state_dict(torch.load(checkpoint_pretrain, weights_only=True)['model_state_dict'])
                
                for name, param in model.named_parameters():
                    if 'layer4' not in name and 'fc' not in name:
                        param.requires_grad = False

                # Fine-tuning
                # for param in model.conv1.parameters():
                #     param.requires_grad = False
                # for param in model.conv2.parameters():
                #     param.requires_grad = False
                # for param in model.conv3.parameters():
                #     param.requires_grad = False


            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = torch.nn.CrossEntropyLoss()
            
            checkpoint_path = f"checkpoint/cnn7_{classes_key}_f{n_fold}.pth"            

            # Training
            print("\nTreining a fault detector...")
            print(type(val_loader))
            htr = train(model, train_loader, val_loader, test_loader, criterion, optimizer,
                num_epochs=num_epochs, device=device, use_class_weights=False,
                checkpoint_path=checkpoint_path, checkpoint_mode="val_loss")

            
            print("\nCarregando melhores pesos...")
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True)['model_state_dict'])
            
            acc = test(model, test_loader, device=device, class_names=[k for k in labels_map.keys()])
            
            history.append((htr, acc))
            accuracies.append(acc)

        total_history.append(history)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        print(f"\nPartial Evaluation Summary")
        print(f"Mean Accuracy: {mean_acc:.4f}")
        print(f"Standard Deviation: {std_acc:.4f}")

        total_accuracy.append(accuracies)

    # Final Summary
    mean_accuracy_by_fold = np.round(np.mean(total_accuracy, axis=0), 4)
    std_accuracy_by_fold = np.round(np.std(total_accuracy, axis=0), 4)

    print("\nFinal Evaluation Summary:")
    print("-------------------------------------------")
    print(f"Mean Accuracy by Fold: {mean_accuracy_by_fold}")
    print(f"Standard Deviation by Fold: {std_accuracy_by_fold}")
    print("-------------------------------------------")
    print(f"\nTotal Accuracy: {np.mean(mean_accuracy_by_fold):.4f}")
    print(f"Standard Deviation: {np.std(total_accuracy):.4f}")
    print("-------------------------------------------")


if __name__ == "__main__":
    
    params = {
        'classes': ['B', 'I'],
        'num_repetitions': 1,
        'learning_rate': 0.0001,
        'num_epochs': 10,
        'model': resnet18,
        'checkpoint_pt': None #"checkpoint/resnet18_uored_bi.pth"
    }
    num_classes = len(params['classes'])
    classes_str = "".join(params['classes'])
    experiment_title = "Resnet18_CWRU_" + classes_str.lower()
    root_dir = "data/spectrograms/cwru_with_outliers"
    history_path = f"pkl_files/{experiment_title}.pkl"
    model_factory = ModelFactory(model_class=params['model'], 
                                 classifier_layer=nn.Sequential(
                                    nn.Dropout(p=0.6),
                                    nn.Linear(512, num_classes)
                                ),
                                 weights= None) #ResNet18_Weights.DEFAULT)

        
    sys.stdout = LoggerWriter(logging.info, experiment_title)
    
    labels_map = {label: index for index, label in enumerate(params['classes'])}

    total_acc = run_kfold(root_dir, params['learning_rate'], params['num_epochs'], params['num_repetitions'], model_factory, 
            labels_map=labels_map, checkpoint_pretrain=params['checkpoint_pt'])
 
        

