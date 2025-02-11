import torch
import torch.nn as nn
import torch.optim as optim


# Loss function VAE (Reconstruction + KL Divergence)
def vae_loss(recon_x, target_x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, target_x, reduction='sum')  # Comparate with target
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  
    return recon_loss + kl_divergence


def train_vae(model, train_loader, val_loader, epochs=100, lr=1e-3, saved_path="saved_models/vae.pth", device="cuda"):
    """
    Train a Variational Autoencoder (VAE) and track validation loss to check for overfitting.
    
    Args:
    - model: VAE model to train
    - train_loader: DataLoader for training set
    - val_loader: DataLoader for validation set
    - epochs: Number of training epochs
    - lr: Learning rate
    - saved_path: Path to save the trained model
    - device: Device for computation ('cuda' or 'cpu')
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Lists to track loss for analysis
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # === Training Phase ===
        model.train()  # Set model to training mode
        train_loss = 0  

        for batch in train_loader:
            input_img, target_img = batch
            input_img, target_img = input_img.to(device), target_img.to(device)

            optimizer.zero_grad()
            recon_img, mu, logvar = model(input_img)  
            loss = vae_loss(recon_img, target_img, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # === Validation Phase ===
        model.eval()  # Set model to evaluation mode (no weight updates)
        val_loss = 0

        with torch.no_grad():  # No gradients needed for validation
            for batch in val_loader:
                input_img, target_img = batch
                input_img, target_img = input_img.to(device), target_img.to(device)

                recon_img, mu, logvar = model(input_img)
                loss = vae_loss(recon_img, target_img, mu, logvar)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Print both losses for analysis
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), saved_path)

    return train_losses, val_losses  # Return losses for further analysis
