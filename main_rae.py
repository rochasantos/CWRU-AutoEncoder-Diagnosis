import torch
from src.dataset import VibrationDataset
from src_rae.rae_classifier import RAEClassifier
from src_rae.rae import RAE
from src_rae.train_rae_classifier import train_rae_classifier
from src_rae.test_rae_classifier import test_rae_classifier
from torch.utils.data import ConcatDataset
from src_rae.finetune import finetune_rae_classifier

if __name__ == "__main__":
    ft_root = "data/processed/f2"
    te_root = "data/processed/cwru/014"

    tr_dataset = ConcatDataset([
        # VibrationDataset("data/processed/f2"),
        VibrationDataset("data/processed/hust"),
        VibrationDataset("data/processed/uored"),
        VibrationDataset("data/processed/paderborn"),
    ])
    ft_dataset = VibrationDataset(ft_root)
    te_dataset = VibrationDataset(te_root)
    
    rae = RAE(input_dim=4200)
    hidden_dim = rae.encoder.hidden_size
    print(f"hidden_dim: {hidden_dim}")
    train_rae_classifier(rae, tr_dataset, num_classes=3, num_epochs=100)
    rae_classifier = RAEClassifier(rae.encoder, hidden_dim, num_classes=3).to(device="cuda")
    rae_classifier.load_state_dict(torch.load("rae_model.pth", weights_only=True))

    # finetune    
    finetune_rae_classifier(rae_classifier=rae_classifier, dataset=ft_dataset, num_epochs=200)

    test_rae_classifier(rae_classifier, te_dataset, ["Normal", "Falha I", "Falha O", "Falha B"])
