import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from src.models import VAE, SpectroNet


def vae_loss(recon, x_original, mu, logvar):
    recon_loss = F.mse_loss(recon, F.interpolate(x_original, size=(480, 480)), reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl


def experimenter():
    # Config
    num_classes = 4
    batch_size = 32
    num_epochs_vae = 20
    num_epochs_spectronet = 10
    saved_vae_path = "saved_models/vae_encoder.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    transform = transforms.ToTensor()
    train_vae_dataset = ConcatDataset([
        datasets.ImageFolder(root="data/spectrogram/cwru/7", transform=transform),
        # datasets.ImageFolder(root="data/spectrogram/cwru/21", transform=transform),
    ])
    train_loader = DataLoader(train_vae_dataset, batch_size=batch_size, shuffle=True)

    # Model
    vae = VAE(reconstruct=False).to(device)
    # optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

    # # Train loop
    # for epoch in range(num_epochs_vae):
    #     vae.train()
    #     total_loss = 0

    #     for x_batch, _ in train_loader:
    #         x_batch = x_batch.to(device)
    #         recon, mu, logvar, z = vae(x_batch)
    #         loss = vae_loss(recon, x_batch, mu, logvar)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item()

    #     print(f"Epoch {epoch+1}/{num_epochs_vae} - Loss: {total_loss:.4f}")

    # # Save encoder
    # torch.save(vae.encoder.state_dict(), saved_vae_path)
    vae.encoder.load_state_dict(torch.load(saved_vae_path, weights_only=True))
    
    # Model
    spectronet_model = SpectroNet(vae.encoder, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(spectronet_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train loop
    for epoch in range(num_epochs_spectronet):
        spectronet_model.train()
        total_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = spectronet_model(x_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs_spectronet} - Loss: {total_loss:.4f}")

    # Save model
    torch.save(spectronet_model.state_dict(), "saved_models/spectronet.pt")
