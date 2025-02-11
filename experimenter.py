import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from models import VariationalAutoencoder, VAE_Classifier
from data_processing import SpectrogramPairDataset
from train_vae import train_vae
from train_classifier import train_classifier
from test_model import test_model


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Experimenter function
def experimenter():
    device="cuda"

    # Hyperparameter configuration
    latent_dim = 16
    batch_size = 32
    num_classes = 4
    saved_vae = "saved_models/vae_300.pth"
    saved_vae_clas = "saved_models/vae_clas_100.pth"

    # Spectrogram transformation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # Directories where the input and output spectrograms are located
    input_dir = "data/spectrograms/007"
    target_dir = "data/spectrograms/021"

    # Loading dataset
    dataset = SpectrogramPairDataset(input_dir, target_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training the VAE
    vae = VariationalAutoencoder(latent_dim=latent_dim).to(device)
    if not os.path.exists(saved_vae):
        print("Starting VAE training.")
        train_vae(vae, dataloader, epochs=350, lr=1e-3, saved_path=saved_vae)
    else:
        print(f"VAE model is already saved in {saved_vae}")

    # Load the weights
    vae = VariationalAutoencoder(latent_dim=16).to(device)
    vae.load_state_dict(torch.load(saved_vae, weights_only=True))
    vae.eval()  # Set to evaluation mode (no training updates)
    vae_classifier = VAE_Classifier(vae, num_classes).to(device)
    
    # Freezes the Encoder weights so they don't change during training
    for param in vae_classifier.encoder.parameters():
        param.requires_grad = False
    
    # Load the datasets
    dataset_7 = ImageFolder("data/spectrograms/007", transform=transform)
    dataset_14 = ImageFolder("data/spectrograms/014", transform=transform)
    dataset_21 = ImageFolder("data/spectrograms/021", transform=transform)
    dataset_clas = ConcatDataset([dataset_7, dataset_21])
    dataloader_clas = DataLoader(dataset_clas, batch_size=batch_size, shuffle=True)
    
    # Start the classifier training
    if not os.path.exists(saved_vae_clas):
        print("Starting VAE_Classifier training.")
        train_classifier(vae_classifier, dataloader_clas, epochs=100, lr=1e-3, saved_path=saved_vae_clas)
    else:
        print(f"VAE_Classifier model is already saved in {saved_vae_clas}")

    # Load the trained VAE classifier
    vae_classifier = VAE_Classifier(vae, num_classes=4).to(device)
    vae_classifier.load_state_dict(torch.load(saved_vae_clas, weights_only=True))

    # Get class names (useful for confusion matrix)
    class_names = dataset_14.classes
    print("Classes:", class_names)
    
    # Evaluate the model
    test_loader = DataLoader(dataset_14, batch_size=32, shuffle=False)
    test_model(vae_classifier, test_loader, class_names)
    
if __name__ == "__main__":
    experimenter()
