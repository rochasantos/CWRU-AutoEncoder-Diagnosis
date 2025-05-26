import sys
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim

from src.utils import LoggerWriter
from src.training import train, test, th
from src.models import ModelFactory, BinaryCNN, HierarchicalClassifier
from src.data_processing import VibrationMapBuilder, VibrationDatasetFromMap


def labels_map(original_labels, group0=[2], group1=[0,1,3]):
    binary_labels = torch.empty_like(original_labels)

    for idx, label in enumerate(original_labels):
        if label.item() in group0:
            binary_labels[idx] = 0
        elif label.item() in group1:
            binary_labels[idx] = 1
        else:
            raise ValueError(f"Label {label.item()} não pertence a nenhum grupo definido.")

    return binary_labels

# Define a list of files to include in the dataset
bearings_files_b7 = [(i, "B") for i in [122, 123, 124, 125]]
bearings_files_b14 = [(i, "B") for i in [189, 190, 191, 192]]
bearings_files_b21 = [(i, "B") for i in [226, 227, 228, 229]]
bearings_files_i7 = [(i, "I") for i in [109, 110, 111, 112]]
bearings_files_i14 = [(i, "I") for i in [174, 175, 176, 177]]
bearings_files_i21 = [(i, "I") for i in [213, 214, 215, 217]]
bearings_files_n7 = [(i, "N") for i in [98]]
bearings_files_n14 = [(i, "N") for i in [99]]
bearings_files_n21 = [(i, "N") for i in [100]]
bearings_files_o7 = [(i, "O") for i in [135, 136, 137, 138]]
bearings_files_o14 = [(i, "O") for i in [201, 202, 203, 204]]
bearings_files_o21 = [(i, "O") for i in [238, 239, 240, 241]]

def get_list_files(key="niob"):

    if key == "BINO":
        return {
            "fold1": {
                "train": [str(b[0]) for b in bearings_files_b14 + bearings_files_b21 + bearings_files_i14 + bearings_files_i21],
                "test": [str(b[0]) for b in bearings_files_b7 + bearings_files_i7]
            },
            "fold2": {
                "train": [str(b[0]) for b in bearings_files_b7 + bearings_files_b21 + bearings_files_i7 + bearings_files_i21],
                "test": [str(b[0]) for b in bearings_files_b14 + bearings_files_i14]
            },
            "fold3": {
                "train": [str(b[0]) for b in bearings_files_b7 + bearings_files_b14 + bearings_files_i7 + bearings_files_i14],
                "test": [str(b[0]) for b in bearings_files_b21 + bearings_files_i21]
            }   
        }

    if key == "BI":
        return {
            "fold1": {
                "train": [str(b[0]) for b in bearings_files_b14 + bearings_files_b21 + bearings_files_i14 + bearings_files_i21],
                "test": [str(b[0]) for b in bearings_files_b7 + bearings_files_i7]
            },
            "fold2": {
                "train": [str(b[0]) for b in bearings_files_b7 + bearings_files_b21 + bearings_files_i7 + bearings_files_i21],
                "test": [str(b[0]) for b in bearings_files_b14 + bearings_files_i14]
            },
            "fold3": {
                "train": [str(b[0]) for b in bearings_files_b7 + bearings_files_b14 + bearings_files_i7 + bearings_files_i14],
                "test": [str(b[0]) for b in bearings_files_b21 + bearings_files_i21]
            }   
        }
    
    if key == "BO":
        return {
            "fold1": {
                "train": [str(b[0]) for b in bearings_files_b14 + bearings_files_b21 + bearings_files_o14 + bearings_files_o21],
                "test": [str(b[0]) for b in bearings_files_b7 + bearings_files_o7]
            },
            "fold2": {
                "train": [str(b[0]) for b in bearings_files_b7 + bearings_files_b21 + bearings_files_o7 + bearings_files_o21],
                "test": [str(b[0]) for b in bearings_files_b14 + bearings_files_o14]
            },
            "fold3": {
                "train": [str(b[0]) for b in bearings_files_b7 + bearings_files_b14 + bearings_files_o7 + bearings_files_o14],
                "test": [str(b[0]) for b in bearings_files_b21 + bearings_files_o21]
            }   
        }
    
    if key == "IO":
        return {
            "fold1": {
                "train": [str(b[0]) for b in bearings_files_i14 + bearings_files_i21 + bearings_files_o14 + bearings_files_o21],
                "test": [str(b[0]) for b in bearings_files_i7 + bearings_files_o7]
            },
            "fold2": {
                "train": [str(b[0]) for b in bearings_files_i7 + bearings_files_i21 + bearings_files_o7 + bearings_files_o21],
                "test": [str(b[0]) for b in bearings_files_i14 + bearings_files_o14]
            },
            "fold3": {
                "train": [str(b[0]) for b in bearings_files_i7 + bearings_files_i14 + bearings_files_o7 + bearings_files_o14],
                "test": [str(b[0]) for b in bearings_files_i21 + bearings_files_o21]
            }   
        }
        

# K-fold cross-validation
def run_kfold(lr, num_epochs, history_path, num_repetitions, model_factory, root_dir, device='cuda'):
    print(f"Model Architecture: {model_factory.build()}")

    total_accuracy = []
    total_history = []

    # Initialize builders for each fold
    builders = {
        f"builder1_f{i+1}_train": VibrationMapBuilder(f"{root_dir}/fold{i+1}/train", list_files=None) for i in range(3) # N/BIO
    }
    builders.update({
        f"builder1_f{i+1}_test": VibrationMapBuilder(f"{root_dir}/fold{i+1}/test", list_files=None) for i in range(3) 
    })
    builders.update({
        f"builder2_f{i+1}_train": VibrationMapBuilder(f"{root_dir}/fold{i+1}/train",
                                                      list_files=get_list_files("BI")[f"fold{i+1}"]["train"]) for i in range(3) # B/I
    })
    builders.update({
        f"builder2_f{i+1}_test": VibrationMapBuilder(f"{root_dir}/fold{i+1}/test", 
                                                     list_files=get_list_files("BI")[f"fold{i+1}"]["test"]) for i in range(3)
        
    })
    builders.update({
        f"builder3_f{i+1}_train": VibrationMapBuilder(f"{root_dir}/fold{i+1}/train",
                                                      list_files=get_list_files("BO")[f"fold{i+1}"]["train"]) for i in range(3)  # N/I
    })
    builders.update({
        f"builder3_f{i+1}_test": VibrationMapBuilder(f"{root_dir}/fold{i+1}/test", 
                                                     list_files=get_list_files("BO")[f"fold{i+1}"]["test"]) for i in range(3) 
        
    })
    builders.update({
        f"builder4_f{i+1}_train": VibrationMapBuilder(f"{root_dir}/fold{i+1}/train",
                                                      list_files=get_list_files("IO")[f"fold{i+1}"]["train"]) for i in range(3)  # N/I
    })
    builders.update({
        f"builder4_f{i+1}_test": VibrationMapBuilder(f"{root_dir}/fold{i+1}/test", 
                                                     list_files=get_list_files("IO")[f"fold{i+1}"]["test"]) for i in range(3) 
        
    })


    for rep in range(num_repetitions):
        print(f"\n------- Repetition: {rep + 1} / {num_repetitions} -------")
        accuracies = []
        history = []

        for n_fold in range(1, 4):  # 3 folds
            print(f"\n------- Fold: {n_fold} -------")

            # Dataset preparation
            train_map1, val_map1 = builders[f"builder1_f{n_fold}_train"].get_split(val_ratio=0.2)
            train_map2, val_map2 = builders[f"builder2_f{n_fold}_train"].get_split(val_ratio=0.2)
            test_map2 = builders[f"builder2_f{n_fold}_test"].get_test_map()
            train_map3, val_map3 = builders[f"builder3_f{n_fold}_train"].get_split(val_ratio=0.2)
            test_map3 = builders[f"builder3_f{n_fold}_test"].get_test_map()
            train_map4, val_map4 = builders[f"builder4_f{n_fold}_train"].get_split(val_ratio=0.2)
            test_map4 = builders[f"builder4_f{n_fold}_test"].get_test_map()
            
            test_map = builders[f"builder1_f{n_fold}_test"].get_test_map()
            
            train_dataset1 = VibrationDatasetFromMap(train_map1, transform=None)
            val_dataset1 = VibrationDatasetFromMap(val_map1, transform=None)
            test_dataset1 = VibrationDatasetFromMap(test_map, transform=None)
            train_dataset2 = VibrationDatasetFromMap(train_map2, transform=None)
            val_dataset2 = VibrationDatasetFromMap(val_map2, transform=None)
            test_dataset2 = VibrationDatasetFromMap(test_map2, transform=None)
            train_dataset3 = VibrationDatasetFromMap(train_map3, transform=None)
            val_dataset3 = VibrationDatasetFromMap(val_map3, transform=None)
            test_dataset3 = VibrationDatasetFromMap(test_map3, transform=None)
            train_dataset4 = VibrationDatasetFromMap(train_map4, transform=None)
            val_dataset4 = VibrationDatasetFromMap(val_map4, transform=None)
            test_dataset4 = VibrationDatasetFromMap(test_map4, transform=None)                   
            
            train_loader1 = DataLoader(train_dataset1, batch_size=32, shuffle=True)
            val_loader1 = DataLoader(val_dataset1, batch_size=32)
            test_loader1 = DataLoader(test_dataset1, batch_size=32)
            train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=True)
            val_loader2 = DataLoader(val_dataset2, batch_size=32)
            test_loader2 = DataLoader(test_dataset2, batch_size=32)
            train_loader3 = DataLoader(train_dataset3, batch_size=32, shuffle=True)
            val_loader3 = DataLoader(val_dataset3, batch_size=32)
            test_loader3 = DataLoader(test_dataset3, batch_size=32)
            train_loader4 = DataLoader(train_dataset4, batch_size=32, shuffle=True)
            val_loader4 = DataLoader(val_dataset4, batch_size=32)
            test_loader4 = DataLoader(test_dataset4, batch_size=32)

            test_dataset = VibrationDatasetFromMap(test_map, transform=None)   
            test_loader = DataLoader(test_dataset, batch_size=32)

            # Model initialization
            model1 = model_factory.build()
            model2 = model_factory.build()
            model3 = model_factory.build()
            model4 = model_factory.build()
            
            # Optionally load pre-trained weights
            # model.load_state_dict(torch.load("checkpoint/binary_uored.pth", weights_only=True)['model_state_dict'])
           
            optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr, weight_decay=1e-5)
            criterion1 = torch.nn.CrossEntropyLoss()
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr, weight_decay=1e-5)
            criterion2 = torch.nn.CrossEntropyLoss()
            optimizer3 = torch.optim.Adam(model3.parameters(), lr=lr, weight_decay=1e-5)    
            criterion3 = torch.nn.CrossEntropyLoss()
            optimizer4 = torch.optim.Adam(model4.parameters(), lr=lr, weight_decay=1e-5)    
            criterion4 = torch.nn.CrossEntropyLoss()

            checkpoint_path1 = f"checkpoint/best_model1_fold{n_fold}.pth"
            checkpoint_path2 = f"checkpoint/best_model2_fold{n_fold}.pth"
            checkpoint_path3 = f"checkpoint/best_model3_fold{n_fold}.pth"
            checkpoint_path4 = f"checkpoint/best_model4_fold{n_fold}.pth"

            # Training
            print("\nTreinando classificador N/BIO...")
            loss_history, accuracy_history, val_loss_history, val_accuracy_history = train(
                model1, train_loader1, val_loader1, criterion1, optimizer1,
                num_epochs=num_epochs, labels_map=labels_map, device=device,
                checkpoint_path=checkpoint_path1, early_stopping=None, scheduler=None
            )
            acc = test(model1, test_loader1, labels_map=labels_map, device=device, class_names=["N", "BIO"])
            print(f"Accuracy N/BIO: {acc:.4f}")

            print("\nTreinando classificador B/I...")
            loss_history, accuracy_history, val_loss_history, val_accuracy_history = train(
                model2, train_loader2, val_loader2, criterion2, optimizer2,
                num_epochs=num_epochs, labels_map=None, device=device,
                checkpoint_path=checkpoint_path2, early_stopping=None, scheduler=None
            )
            acc = test(model2, test_loader2, labels_map=labels_map, device=device, class_names=["B", "I"])
            print(f"Accuracy B/I: {acc:.4f}")
            
            print("\nTreinando classificador B/O...")
            loss_history, accuracy_history, val_loss_history, val_accuracy_history = train(
                model3, train_loader3, val_loader3, criterion3, optimizer3,
                num_epochs=num_epochs, labels_map=None, device=device,
                checkpoint_path=checkpoint_path3, early_stopping=None, scheduler=None
            )
            acc = test(model3, test_loader3, labels_map=labels_map, device=device, class_names=["B", "O"])
            print(f"Accuracy B/O: {acc:.4f}")

            print("\nTreinando classificador I/O...")
            loss_history, accuracy_history, val_loss_history, val_accuracy_history = train(
                model4, train_loader4, val_loader4, criterion4, optimizer4,
                num_epochs=num_epochs, labels_map=None, device=device,
                checkpoint_path=checkpoint_path4, early_stopping=None, scheduler=None
            )
            acc = test(model4, test_loader4, labels_map=labels_map, device=device, class_names=["I", "O"])
            print(f"Accuracy I/O: {acc:.4f}")

            print("\nCarregando melhores pesos...")
            model1.load_state_dict(torch.load(checkpoint_path1, weights_only=True)['model_state_dict'])
            model2.load_state_dict(torch.load(checkpoint_path2, weights_only=True)['model_state_dict'])
            model3.load_state_dict(torch.load(checkpoint_path3, weights_only=True)['model_state_dict'])
            model4.load_state_dict(torch.load(checkpoint_path4, weights_only=True)['model_state_dict'])

            print("\nInicializando classificador hierárquico...")
            hierarchical_model = HierarchicalClassifier(clf_n_bio=model1, 
                                                        clf_b_i=model2, 
                                                        clf_b_o=model3, 
                                                        clf_i_o=model4).to(device)
            
            print("\nExecutando validação hierárquica...")
            acc = th(hierarchical_model, test_loader, device)
            # Aqui deve ser colocado o classificador hierárquico de forma que ele possa receber como parâmetro os classificadores binários.

            history.append((loss_history, accuracy_history, val_loss_history, val_accuracy_history, acc))
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
    # Setup Logger
    experiment_title = "lstm_005"
    root_dir = "data/processed/cwru"
    history_path = f"pkl_files/{experiment_title}.pkl"
    # sys.stdout = LoggerWriter(logging.info, experiment_title)

    # Parameters
    num_repetitions = 1
    lr = 0.0001
    num_epochs = 25

    model_factory = ModelFactory(model_class=BinaryCNN, input_channels=1, input_length=4096)

    run_kfold(lr, num_epochs, history_path, num_repetitions, model_factory, root_dir)
