import torch
import torch.nn as nn

class RAEClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes):
        super(RAEClassifier, self).__init__()
        self.encoder = encoder  

        # 🔹 Congelar os pesos do encoder para evitar re-treinamento
        for param in self.encoder.parameters():
            param.requires_grad = False  

        # 🔹 Ajuste para garantir que a camada Linear recebe `hidden_dim=64`
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),  # ✅ Agora recebe 64 como entrada!
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape  

        with torch.no_grad():
            _, (h_n, _) = self.encoder(x)  # 🔹 Pegamos o último estado do LSTM
            encoded = h_n[-1]  # `encoded.shape = (batch_size, hidden_dim)`

        # print(f"✅ Shape de encoded antes do classificador: {encoded.shape}")  # Deve ser (batch_size, 64)

        logits = self.classifier(encoded)  
        return logits
