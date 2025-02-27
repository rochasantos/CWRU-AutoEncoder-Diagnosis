import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from src.dataset import VibrationDataset
from src.test_classifier_vae import test_classifier


class VibrationDatasetVAE(Dataset):
    def __init__(self, input_root_dir, output_root_dir, transform=None):
        """
        input_root_dir: Diretório contendo os sinais de entrada.
        output_root_dir: Diretório contendo os sinais de saída esperados.
        transform: Transformações opcionais aplicadas aos sinais.
        """
        self.input_root_dir = input_root_dir
        self.output_root_dir = output_root_dir
        self.transform = transform
        self.input_files, self.output_files = self._load_file_paths()  # Carrega os caminhos

    def _load_file_paths(self):
        """
        Percorre os diretórios de entrada e saída e associa os arquivos correspondentes.
        """
        input_files = []
        for class_dir in os.listdir(self.input_root_dir):
            full_class_dir = os.path.join(self.input_root_dir, class_dir)
            if os.path.isdir(full_class_dir):
                for filename in os.listdir(full_class_dir):
                    if filename.endswith(".npy"):
                        input_files.append(os.path.join(full_class_dir, filename))

        output_files = []
        for class_dir in os.listdir(self.output_root_dir):
            full_class_dir = os.path.join(self.output_root_dir, class_dir)
            if os.path.isdir(full_class_dir):
                for filename in os.listdir(full_class_dir):
                    if filename.endswith(".npy"):
                        output_files.append(os.path.join(full_class_dir, filename))

        max_length = len(input_files) if len(input_files)<len(output_files) else len(output_files)
        return input_files[:max_length], output_files[:max_length]
            

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        """
        Carrega os sinais de entrada e saída a partir dos arquivos numpy correspondentes.
        """
        input_file = self.input_files[idx]
        output_file = self.output_files[idx]

        input_signal = np.load(input_file)[:-1]  # Carrega sinal de entrada
        output_signal = np.load(output_file)[:-1]  # Carrega sinal de saída esperado

        # Convertendo para tensor (1D CNN espera [C, L])
        input_signal = torch.tensor(input_signal, dtype=torch.float32).unsqueeze(0)
        output_signal = torch.tensor(output_signal, dtype=torch.float32).unsqueeze(0)

        # Aplicando transformações, se houver
        if self.transform:
            input_signal = self.transform(input_signal)
            output_signal = self.transform(output_signal)

        return input_signal, output_signal


# 🔹 Definição do modelo VAE
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Média do espaço latente
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log da variância

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Converte logvar para desvio padrão
        eps = torch.randn_like(std)  # Amostragem de ruído
        return mu + eps * std  # Amostra do espaço latente

    def forward(self, x):
        batch_size, _, signal_length = x.shape  # Pegando o tamanho real do sinal
        x = x.view(batch_size, -1)  # Achata para (batch_size, input_dim)
        
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)  # Amostra do espaço latente
        
        reconstructed = self.decoder(z)
        reconstructed = reconstructed.view(batch_size, 1, signal_length)  # Volta ao formato (B, 1, L)
        return reconstructed, mu, logvar

# 🔹 Função de perda (Reconstrução + KL Divergence)
def loss_function(reconstructed, y, mu, logvar):
    recon_loss = nn.MSELoss()(reconstructed, y)  # Erro de reconstrução
    kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence
    return recon_loss + kl_divergence


class VAEClassifier(nn.Module):
    def __init__(self, encoder, input_dim, num_classes):
        super(VAEClassifier, self).__init__()
        self.encoder = encoder  # Reutiliza o encoder do VAE pré-treinado

        # 🔹 Congelar os pesos do encoder para evitar treinamento
        for param in self.encoder.parameters():
            param.requires_grad = False  # 🚨 Isso impede o treinamento do encoder!

        # 🔹 Camadas adicionais para classificação
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        batch_size, _, signal_length = x.shape
        x = x.view(batch_size, -1)  # Achata para (batch_size, input_dim)

        with torch.no_grad():  # 🔹 Garante que o encoder não acumule gradientes
            encoded = self.encoder(x)
            # print(f"🔍 Shape de encoded: {encoded.shape}")  

        logits = self.classifier(encoded)  # 🔹 Apenas o classificador será treinado
        return logits


# 🔹 Configuração do treinamento
def train_vae(input_root, output_root, num_epochs=50, batch_size=32, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 Criando o Dataset e DataLoader
    dataset = VibrationDatasetVAE(input_root, output_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 🔹 Criando o modelo e otimizador
    input_dim = dataset[0][0].shape[-1]  # Obtendo o tamanho real do sinal
    vae = VAE(input_dim=input_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    # 🔹 Loop de Treinamento
    for epoch in range(num_epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            reconstructed, mu, logvar = vae(x_batch)
            loss = loss_function(reconstructed, y_batch, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}')
    
    torch.save(vae.state_dict(), "vae_model.pth")
    print("✅ Treinamento concluído!")

    return vae  # Retorna o modelo treinado

# 🔹 Ajuste fino e treinamento do classificador
def train_classifier(vae, dataset_path, num_classes, num_epochs=30, batch_size=32, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VibrationDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    latent_dim = vae.fc_mu.out_features
    classifier = VAEClassifier(vae.encoder, 64, num_classes=3).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for x_batch, labels in dataloader:
            x_batch, labels = x_batch.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = classifier(x_batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}, Acc: {acc:.2f}%')
    
    torch.save(classifier.state_dict(), "vae_classifier_model.pth")
    print("✅ Classificador Treinado!")


if __name__ == "__main__":
    input_root, output_root = "data/processed/hust", "data/processed/paderborn"
    dataset = VibrationDatasetVAE(input_root, output_root)
    te_root = "data/processed/cwru/014"
    te_dataset = VibrationDataset(te_root)
    ft_root = "data/processed/f2"
    ft_dataset = VibrationDataset(ft_root)
    
    # train vae
    train_vae(input_root, output_root, num_epochs=100)

    # finetune
    vae = VAE(input_dim=4200)
    vae.load_state_dict(torch.load("vae_model.pth", weights_only=True))
    train_classifier(vae, ft_root, num_classes=3, num_epochs=100, learning_rate=0.01)
    
    # test
    classifier = VAEClassifier(vae.encoder, 64, num_classes=3).to(device="cuda")
    classifier.load_state_dict(torch.load("vae_classifier_model.pth", weights_only=True))
    test_classifier(classifier, test_dataset=te_dataset, class_names=["I", "O", "B"])