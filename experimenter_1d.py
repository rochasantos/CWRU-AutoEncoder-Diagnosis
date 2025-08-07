import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------- Dataset Definition ----------------
class BearingDataset(Dataset):
    def __init__(self, root_dir, severities):
        self.samples = []
        self.labels = []
        for severity in severities:
            for label in ['I', 'B']:
                path = os.path.join(root_dir, severity, label, '*.npy')
                files = glob.glob(path)
                self.samples.extend(files)
                self.labels.extend([label] * len(files))
        self.encoder = LabelEncoder()
        self.labels = self.encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        signal = np.load(self.samples[idx])[:3930].astype(np.float32).flatten()
        label = self.labels[idx]
        return torch.tensor(signal).unsqueeze(0), torch.tensor(label).long()


# ---------------- CNN-1D Model ----------------
class CNN1D(nn.Module):
    def __init__(self, num_classes=2, n_conv_layers=2, filters=32, kernel_size=32, pool_size=8):
        super(CNN1D, self).__init__()
        layers = []
        in_channels = 1
        for i in range(n_conv_layers):
            layers.append(nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pool_size))
            in_channels = filters
        self.conv = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters * ((3930 // (pool_size ** n_conv_layers))), 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x


# ---------------- Training & Evaluation Functions ----------------
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=20):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss.item():.4f}")



def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['B', 'I']))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nAccuracy: {acc:.4f}")
    
    return acc



# ---------------- Main Execution ----------------
if __name__ == "__main__":
    # Parameters
    root_dir = 'cwru'  # update to 'cwru_aug' if needed
    train_severities = ['007', '021']
    test_severities = ['014']
    batch_size = 64
    learning_rate = 0.0001
    num_epochs = 20
    repetitions = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accuracies = []
    for rep in range(repetitions):
        print(f"Repetition: {rep}")
        # Load datasets
        train_dataset = ConcatDataset([
            BearingDataset("data/processed/cwru_aug", train_severities),
            BearingDataset("data/processed/cwru", train_severities),
        ])
        test_dataset = BearingDataset("data/processed/cwru_t", test_severities)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Model, loss, optimizer
        model = CNN1D(n_conv_layers=2, filters=32, kernel_size=32)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train and evaluate
        train_model(model, train_loader, criterion, optimizer, device, num_epochs)
        accuracy = evaluate_model(model, test_loader, device)
        accuracies.append(accuracy)
    
    accuracies = np.array(accuracies)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    print(f"\nModel Evaluation Summary:")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation: {std_accuracy:.4f}")