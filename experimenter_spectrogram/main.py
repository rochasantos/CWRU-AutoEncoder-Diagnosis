import torch
from torch.utils.data import ConcatDataset
from .model import RAE
from .train import train_rae
from .train_classifier import RAEClassifier
from .finetune import finetune_rae_classifier
from .test import test_rae_classifier
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


def experimenter():
    
    tr_num_epochs = 100
    tr_lr = 0.0001
    ft_num_epochs = 10
    ft_lr = 0.001
    
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

    repetition = 5
    for rep in range(repetition):
        print(f"Repetition {rep+1}/{repetition}")
        for i in [0, 1, 2]:
            fold = i + 1
            print(f"Fold: {fold}")
            save_model = f"saved_models/rae1_f{fold}.pth"
            save_cls = f"saved_models/rae_cls1_f{fold}.pth"
            save_best_model_path = f"saved_models/best_rae1_f{fold}.pth"

            datasets_copy = datasets[:]
            te_dataset = datasets_copy.pop(i)
            tr_dataset = ConcatDataset(datasets_copy)    
            ft_dataset = ConcatDataset(datasets_copy[:2])

            print(f" Training datasets: {[ds.root for ds in datasets_copy]}")
            print(f" Fine-tuning datasets: {[ds.root for ds in datasets_copy[:2]]}")
            print(f" Testing dataset: {te_dataset.root}")
            
            # train
            rae = RAE()
            train_rae(rae, tr_dataset, num_epochs=tr_num_epochs, learning_rate=tr_lr, freeze_layers_idx=[0, 1, 2, 3], save_path=save_model)

            # finetune
            rae.load_state_dict(torch.load(save_model, weights_only=True))  # Carrega os pesos salvos
            rae.eval()    
            classifier = RAEClassifier(rae.encoder, num_classes=3)
            finetune_rae_classifier(classifier, ft_dataset, te_dataset, num_epochs=ft_num_epochs, learning_rate=ft_lr, save_path=save_cls,
                                    save_best_model_path=save_best_model_path, eary_stopping_enabled=False)

            # test
            classifier.load_state_dict(torch.load(save_cls, weights_only=True))  # Carrega os pesos salvos
            test_rae_classifier(classifier, te_dataset, class_names)
        