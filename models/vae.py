import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = None  # Será definido dinamicamente
        self.fc_logvar = None  # Será definido dinamicamente
        self.fc_decode = None  # Será definido dinamicamente

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Achatar dinamicamente
        
        if self.fc_mu is None:
            self.fc_mu = nn.Linear(x.shape[1], self.latent_dim).to(x.device)
            self.fc_logvar = nn.Linear(x.shape[1], self.latent_dim).to(x.device)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        if self.fc_decode is None:
            self.fc_decode = nn.Linear(self.latent_dim, 128 * 16 * 16).to(z.device)
        
        z = self.fc_decode(z)
        z = z.view(-1, 128, 16, 16)  # Ajusta automaticamente
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
