import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from src.models.rae import RAE
from src.training.train import train_rae
from src.models.rae_classifier import RAEClassifier
from src.training.finetune import finetune_rae_classifier
from src.training.test import test_rae_classifier


def experimenter():
    
    tr_num_epochs = 200
    tr_lr = 0.0001
    ft_num_epochs = 40
    ft_lr = 0.001
    repetition = 5
    
    class_names = ["I", "O", "B"]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 

    datasets = [
        ImageFolder("data/spectrogram/cwru_7", transform=transform),
        ImageFolder("data/spectrogram/cwru_14", transform=transform),
        ImageFolder("data/spectrogram/cwru_21", transform=transform),
        ImageFolder("data/spectrogram/uored", transform=transform),
        ImageFolder("data/spectrogram/hust", transform=transform)
    ]

    print("Experimenter with RAE\n")
    print("Parameters")
    print("------------")
    print(" Training")
    print(f" * Number of epochs: {tr_num_epochs}")
    print(f" * Learning rate: {tr_lr}")
    print(" Fine-tuning")
    print(f" * Number of epochs: {tr_num_epochs}")
    print(f" * Learning rate: {tr_lr}")
    print("----------------------------------------------")

    total_accuracy = []
    for rep in range(repetition):
        print(f"Repetition {rep+1}/{repetition}")
        accuracies = []
        for i in [1,2,3]:
            fold = i + 1
            print(f"Fold {fold}")
            save_model = f"saved_models/rae1_f{fold}_rep{rep}.pth"
            save_cls = f"saved_models/rae_cls1_f{fold}_rep{rep}.pth"
            save_best_model_path = f"saved_models/best_rae1_f{fold}_rep{rep}.pth"

            datasets_copy = datasets[:]
            te_dataset = datasets_copy.pop(i)
            tr_dataset = ConcatDataset(datasets_copy)    
            ft_dataset = ConcatDataset(datasets_copy[:2])
            val_dataset = te_dataset

            print(f" Training datasets: {[ds.root for ds in datasets_copy]}")
            print(f" Fine-tuning datasets: {[ds.root for ds in datasets_copy[:2]]}")
            print(f" Testing dataset: {te_dataset.root}")
            
            # train
            rae = RAE()
            train_rae(rae, tr_dataset, num_epochs=tr_num_epochs, learning_rate=tr_lr, freeze_layers_idx=[0, 1, 2, 3], save_path=save_model)

            # finetune
            # rae.load_state_dict(torch.load(save_model, weights_only=True))  # Carrega os pesos salvos
            rae.eval()    
            classifier = RAEClassifier(rae.encoder, num_classes=3)
            finetune_rae_classifier(classifier, ft_dataset, val_dataset, num_epochs=ft_num_epochs, learning_rate=ft_lr, save_path=save_cls,
                                    save_best_model_path=save_best_model_path, eary_stopping_enabled=True)

            # test
            # classifier.load_state_dict(torch.load(save_cls, weights_only=True))  # Carrega os pesos salvos
            accuracy, cm = test_rae_classifier(classifier, te_dataset, class_names)
            print(f"\n🎯 Accuracy on the test set: {accuracy:.2f}%\n")
            print("📊 Confusion Matrix:")
            print(cm)
            accuracies.append(accuracy)
        total_accuracy.append(accuracies)
    mean_accuracy = np.mean(total_accuracy, axis=0)
    std = np.std(total_accuracy, axis=0)
    for i, accuracy in enumerate(mean_accuracy, start=1):
        print(f'Mean accuracy for fold {i}: {accuracy:.2f}%, Std: {np.round(std[i-1], 2)}%')