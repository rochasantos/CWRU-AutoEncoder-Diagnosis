import sys
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch import optim

from src.utils import LoggerWriter
from src.training import train, test, th
from src.models import ModelFactory, CNN1, CNN7, CNN8, ADCNN1D, CNN10
from src.data_processing import VibrationDataset

from src.data_processing.data_augmentation import TransformDataAugmentation


transform = TransformDataAugmentation()

# K-fold cross-validation
def run_kfold(root_dir, lr, num_epochs, history_path, num_repetitions, model_factory, 
              labels_map, use_class_weights=False, checkpoint_pretrain=None, device='cuda'):
    print(f"Model Architecture: {model_factory.build()}")

    classes_key = "".join(labels_map.keys()).lower()

    total_accuracy = []
    total_history = []

    for rep in range(num_repetitions):
        print(f"\n------- Repetition: {rep + 1} / {num_repetitions} -------")
        accuracies = []
        history = []

        for n_fold in range(1, 4):  # 3 folds
            fold = f"fold{n_fold}"
            print(f"\n------- Fold: {n_fold} -------")

            # CWRU Augmentated
            cwru_aug_dataset = [
                VibrationDataset(f"data/processed/cwru_{classes_key}_noise_4096/{fold}/train", labels_map=labels_map),
                VibrationDataset(f"data/processed/cwru_{classes_key}_permute_4096/{fold}/train", labels_map=labels_map),
                VibrationDataset(f"data/processed/cwru_{classes_key}_local_reversing_4096/{fold}/train", labels_map=labels_map),
                VibrationDataset(f"data/processed/cwru_{classes_key}_local_random_reversing_4096/{fold}/train", labels_map=labels_map),
                VibrationDataset(f"data/processed/cwru_{classes_key}_global_reversing_4096/{fold}/train", labels_map=labels_map),
                VibrationDataset(f"data/processed/cwru_{classes_key}_local_splicing_4096/{fold}/train", labels_map=labels_map),
                VibrationDataset(f"data/processed/cwru_{classes_key}_local_zooming_4096/{fold}/train", labels_map=labels_map),
                VibrationDataset(f"data/processed/cwru_{classes_key}_global_zooming_4096/{fold}/train", labels_map=labels_map)
            ]
            train_dataset = ConcatDataset([
                VibrationDataset(f"{root_dir}/{fold}/train", labels_map=labels_map),
                # VibrationDataset(f"data/processed/uored_{classes_key}", labels_map=labels_map),
                # VibrationDataset(f"data/processed/hust_6205_{classes_key}", labels_map=labels_map),
                *cwru_aug_dataset,
            ])

            val_dataset = VibrationDataset(f"{root_dir}/{fold}/val", labels_map=labels_map)
            test_dataset = VibrationDataset(f"{root_dir}/{fold}/test", labels_map=labels_map)                         
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)
                       
            # Model initialization
            model = model_factory.build()           
            
            if checkpoint_pretrain:
                print("Starting fine-tuning.")
                # Optionally load pre-trained weights
                model.load_state_dict(torch.load(checkpoint_pretrain, weights_only=True)['model_state_dict'])

                # Fine-tuning
                for param in model.conv1.parameters():
                    param.requires_grad = False
                for param in model.conv2.parameters():
                    param.requires_grad = False
                for param in model.conv3.parameters():
                    param.requires_grad = False

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=lr
            ) #, weight_decay=1e-5)
            criterion = torch.nn.CrossEntropyLoss()
            
            checkpoint_path = f"checkpoint/clf_f{n_fold}.pth"            

            # Training
            print("\nTreining a fault detector...")
            htr = train(model, train_loader, val_loader, test_loader, criterion, optimizer,
                num_epochs=num_epochs, device=device, use_class_weights=False,
                checkpoint_path=checkpoint_path, checkpoint_mode="val_accuracy")
            
            print("\nCarregando melhores pesos...")
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True)['model_state_dict'])
            
            acc = test(model, test_loader, device=device, class_names=[k for k in sorted(labels_map.keys())])
            
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

    # Save history
    with open(history_path, "wb") as f:
        pickle.dump([total_history, total_accuracy], f)


if __name__ == "__main__":
    
    # Parameters
    classes = sorted(['I', 'O'])
    checkpoint_pretrain = "checkpoint/cnn10_paderborn_io.pth"
    num_repetitions = 5
    lr = 1e-5
    num_epochs = 20
    model = CNN10

    # Setup Logger
    classes_str = ''.join(classes)
    experiment_title = f"{model.__name__}_{classes_str}"
    root_dir = f"data/processed/cwru_{classes_str.lower()}"
    history_path = f"pkl_files/{experiment_title}.pkl"
    sys.stdout = LoggerWriter(logging.info, experiment_title)


    labels_map = {label: index for index, label in enumerate(classes)}
    model_factory = ModelFactory(model_class=model, input_length=4096, num_classes=2)

    run_kfold(root_dir, lr, num_epochs, history_path, num_repetitions, model_factory, 
              labels_map=labels_map, checkpoint_pretrain=checkpoint_pretrain)
