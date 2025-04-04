import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=128, reconstruct=True):
        super().__init__()
        self.reconstruct = reconstruct
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # (3, 480, 480) → (32, 240, 240)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # → (64, 120, 120)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # → (128, 60, 60)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# → (256, 30, 30)
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),# → (512, 15, 15)
            nn.ReLU(),
        )

        self.flattened_size = 512 * 15 * 15  # = 115200

        self.fc_mu     = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)

        if self.reconstruct:
            self.decoder = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),  # (15 → 30)
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),  # (30 → 60)
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),  # (60 → 120)
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),  # (120 → 240)
                nn.Conv2d(32, 3, 3, 1, 1),
                nn.Upsample(scale_factor=2),  # (240 → 480)
                nn.Sigmoid()  # Normalização entre 0 e 1
            )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        encoded = self.encoder(x)
        encoded_flat = encoded.view(batch_size, -1)

        mu = self.fc_mu(encoded_flat)
        logvar = self.fc_logvar(encoded_flat)
        z = self.reparameterize(mu, logvar)

        z = self.fc_decode(z)
        z = z.view(batch_size, 512, 15, 15)

        if self.reconstruct:
            recon = self.decoder(z)
            return recon, mu, logvar, z
        else:
            return z, mu, logvar
