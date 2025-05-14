import torch
import torch.nn as nn
import torch.nn.functional as F

class ArticleCNN1D(nn.Module):
    def __init__(self, input_length=9600, num_classes=2):
        super(ArticleCNN1D, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, stride=2, padding=1),  # BC1
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            nn.Conv1d(4, 8, kernel_size=3, stride=2, padding=1),  # BC2
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            # nn.Conv1d(8, 12, kernel_size=3, stride=2, padding=1),  # BC3
            # nn.BatchNorm1d(12),
            # nn.LeakyReLU(),
            # nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            # nn.Conv1d(12, 16, kernel_size=3, stride=2, padding=1),  # BC4
            # nn.BatchNorm1d(16),
            # nn.LeakyReLU(),
            # nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        conv_output_size = self._get_conv_output(input_length)  # ou o tamanho real da entrada
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(conv_output_size, num_classes)
        )

    def _get_conv_output(self, input_length=9600):
        with torch.no_grad():
            x = torch.zeros(1, 1, input_length)  # Simula entrada com tamanho real
            x = self.features(x)                # Passa pela sequência convolucional
            return x.view(1, -1).shape[1]       # Flatten e obtém tamanho final

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
