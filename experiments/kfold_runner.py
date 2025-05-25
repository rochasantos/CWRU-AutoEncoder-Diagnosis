import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from src.training import train, test
from src.data_processing import VibrationMapBuilder, VibrationDatasetFromMap

# K-fold cross-validation
def kfold_runner(root_dir, learning_rate, num_epochs, batch_size, 
                 num_repetitions, model_factory, experiment_key, title, device='cuda'):

    print(f"Model: {model_factory.build()}")

    total_accuracy = []
    total_history = []

    builders = {
        "builder_f1_train": VibrationMapBuilder(f"{root_dir}/fold1/train"),
        "builder_f2_train": VibrationMapBuilder(f"{root_dir}/fold2/train"),
        "builder_f3_train": VibrationMapBuilder(f"{root_dir}/fold3/train"),
        "builder_f1_test": VibrationMapBuilder(f"{root_dir}/fold1/test"),
        "builder_f2_test": VibrationMapBuilder(f"{root_dir}/fold2/test"),
        "builder_f3_test": VibrationMapBuilder(f"{root_dir}/fold3/test"),
    }

    for rep in range(num_repetitions):
        print(f"\n------- Repetition: {rep+1} / {num_repetitions} -------")
        accuracies = []
        history = []
        for i in range(3): # 3 folds
            n_fold = i+1
            print(f"\n------- Fold: {n_fold} -------")
            
            train_map, val_map = builders[f"builder_f{n_fold}_train"].get_split(val_ratio=0.2)
            test_map = builders[f"builder_f{n_fold}_test"].get_test_map()
            
            train_dataset = VibrationDatasetFromMap(train_map)
            val_dataset = VibrationDatasetFromMap(val_map)            
            test_dataset = VibrationDatasetFromMap(test_map)       

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            # Starting train            
            model = model_factory.build()
            model.load_state_dict(torch.load("checkpoint/cnn13_uored.pth", weights_only=True)['model_state_dict'])

            for name, param in model.named_parameters():
                if "conv" in name:
                    param.requires_grad = False
                
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
            criterion = torch.nn.CrossEntropyLoss()
            checkpoint_path = f"checkpoint/best_model_{n_fold}.pth"
            loss_history, accuracy_history, val_loss_history, val_accuracy_history = train(model, train_loader, val_loader, criterion, optimizer, 
                            num_epochs=num_epochs, device=device, checkpoint_path=checkpoint_path, early_stopping=None ) #EarlyStopping(patience=5, start_threshold=0.06, min_delta=0.005)

            # Avaliação final no conjunto de teste
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True)['model_state_dict'])
            acc = test(model, test_loader, device=device, class_names=["B", "I"])
            
            history.append((loss_history, accuracy_history, val_loss_history, val_accuracy_history, acc))
            accuracies.append(acc)

        total_history.append(history)
        print(f"\nParcial Evaluation Summary")
        print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
        print(f"Standard Deviation: {np.std(accuracies):.4f}")
        total_accuracy.append(accuracies)
    print("\nFinal Evaluation Summary:")
    print("-------------------------------------------")
    print(f"Mean Accuracy by Fold: {np.round(np.mean(total_accuracy, axis=0), 4)}")
    print(f"Standard Deviation by Fold: {np.round(np.std(total_accuracy, axis=0), 4)}")
    print("-------------------------------------------")
    print(f"\nTotal Accuracy: {np.mean(np.mean(total_accuracy, axis=1)):.4f}")
    print(f"Standard Deviation: {np.std(total_accuracy):.4f}")
    print("-------------------------------------------")
    
    history_path = f"pkl_files/{experiment_key}.pkl"
    with open(history_path, "wb") as f:
        pickle.dump([total_history, total_accuracy], f)
