import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class RAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(RAE, self).__init__()

        # 🔹 Encoder baseado na ResNet18
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove a última camada FC

        # 🔹 Espaço latente
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # 🔹 Decoder
        self.decoder_fc = nn.Linear(latent_dim, 512 * 7 * 7)  # Projeta para feature map inicial

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 🔹 Agora ajustado
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)

        # 🔹 Encoder
        encoded = self.encoder(x).view(batch_size, -1)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)

        # 🔹 Decoder
        z = self.decoder_fc(z).view(batch_size, 512, 7, 7)  # 🔹 Ajuste inicial correto
        reconstructed = self.decoder(z)  # 🔹 Agora a saída deve ser [batch_size, 3, 224, 224]

        return reconstructed, mu, logvar


class VAE(nn.Module):  # 🔹 Alterei o nome para VAE
    def __init__(self, latent_dim=256):
        super(VAE, self).__init__()

        # 🔹 Encoder baseado na ResNet18
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove a última camada FC

        # 🔹 Espaço latente
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # 🔹 Decoder
        self.decoder_fc = nn.Linear(latent_dim, 512 * 7 * 7)  # Projeta para feature map inicial

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 🔹 Agora ajustado
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)

        # 🔹 Encoder
        encoded = self.encoder(x).view(batch_size, -1)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)

        # 🔹 Decoder
        z = self.decoder_fc(z).view(batch_size, 512, 7, 7)  # 🔹 Ajuste inicial correto
        reconstructed = self.decoder(z)  # 🔹 Agora a saída deve ser [batch_size, 3, 224, 224]

        return reconstructed, mu, logvar  # 🔹 Retorna os parâmetros do espaço latente
