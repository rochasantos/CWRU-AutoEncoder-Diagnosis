import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=5):
        super(CNN1D, self).__init__()
        
        # Basic Convolutional (BC) modules
        self.bc1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.bc2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.bc3 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=12, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(12),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.bc4 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Fully Connected (FC) Module
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2640, num_classes)  # Assumindo que o tamanho da entrada é 10k e a saída da CNN é B×16×40
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        x = self.bc1(x)
        x = self.bc2(x)
        x = self.bc3(x)
        x = self.bc4(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
