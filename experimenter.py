import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from models import VariationalAutoencoder, VAE_Classifier
from data_processing import SpectrogramPairDataset
from scripts.train_vae import train_vae
from scripts.train_classifier import train_classifier
from scripts.test_vae import test_vae


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Experimenter function
def experimenter():
    device="cuda"

    # Hyperparameter configuration
    latent_dim = 32
    num_classes = 4
    saved_vae = "saved_models/vae.pth"
    saved_vae_clas = "saved_models/vae_cl.pth"

    # Spectrogram transformation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Directories where the input and output spectrograms are located
    dir_7 = "data/temp/007"
    dir_14 = "data/spectrograms/014"
    dir_21 = "data/spectrograms/021"

    # Loading dataset
    dataset_tr = ImageFolder(root=dir_7, transform=transform)
    dataloader1 = DataLoader(dataset_tr, batch_size=64, shuffle=False)
    
    # Training the VAE
    vae = VariationalAutoencoder(latent_dim=latent_dim).to(device)
    print("Starting VAE training.")
    train_vae(vae, dataloader1, dataloader1, epochs=150, lr=1e-2, saved_path=saved_vae)

    # Load the weights
    vae.eval()  # Set to evaluation mode (no training updates)
    vae_classifier = VAE_Classifier(vae, latent_dim, num_classes).to(device)
    
    # Freezes the Encoder weights so they don't change during training
    for param in vae_classifier.encoder.parameters():
        param.requires_grad = False
    
    # Load the datasets
    dataset_te = ConcatDataset([ImageFolder(dir_7, transform=transform),
                   ImageFolder(dir_14, transform=transform),
                   ImageFolder(dir_21, transform=transform)])
    dataloader_te = DataLoader(dataset_tr, batch_size=64, shuffle=False)
    
    # Start the classifier training
    print("Starting VAE_Classifier training.")
    train_classifier(vae_classifier, dataloader_te, epochs=100, lr=1e-4, saved_path=saved_vae_clas)

    # Get class names (useful for confusion matrix)
    class_names = dataset_tr.classes
    print("Classes:", class_names)
    
    # Evaluate the model
    test_loader = DataLoader(dataset_tr, batch_size=64, shuffle=False)
    test_vae(vae_classifier, test_loader, class_names)
    
if __name__ == "__main__":
    experimenter()
