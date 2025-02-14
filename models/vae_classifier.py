import torch
import torch.nn as nn
import torch.optim as optim

# Definition of the classification model with VAE
class VAE_Classifier(nn.Module):
    def __init__(self, vae_encoder, latent_dim, num_classes=4):  # Número de classes pode variar
        super(VAE_Classifier, self).__init__()

        # VAE Encoder
        self.encoder = vae_encoder

        # full connected layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),  # 16 is the dimension of the latent space of the VAE
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)  
        )

    def forward(self, x):
        mu, logvar = self.encoder.encode(x)  # Feature extraction
        z = self.encoder.reparameterize(mu, logvar)  # Latent vector
        return self.classifier(z)
