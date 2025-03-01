import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.training.early_stopping import EarlyStopping


def vae_loss_function(reconstructed, x, mu, logvar):
    recon_loss = nn.MSELoss()(reconstructed, x)  # 🔹 Erro de reconstrução
    kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # 🔹 KL Divergence
    return recon_loss + 0.1 * kl_divergence  # 🔹 O peso 0.1 pode ser ajustado para regularização

def train_rae(rae, dataset, num_epochs=50, batch_size=32, learning_rate=1e-3, 
              save_path="rae_spectro_model.pth",  freeze_layers_idx=None, device="cuda", 
              eary_stopping_enabled=False, save_best_model_path="best_train_model.pth"):
    """
    Treina o modelo RAE permitindo escolher quais camadas da ResNet18 serão congeladas.

    Args:
        rae: Modelo RAE a ser treinado.
        dataset: Dataset de treino.
        num_epochs: Número de épocas.
        batch_size: Tamanho do batch.
        learning_rate: Taxa de aprendizado.
        freeze_layers_idx: Lista de índices das camadas da ResNet18 a serem congeladas. Se None, todas são treináveis.
        device: "cuda" ou "cpu".
    """
    print(f"Starting training...")
    print(f"Learning rate: {learning_rate}, number of epochs: {num_epochs}")
    rae.to(device)
    rae.train()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 🔹 Se for fornecida uma lista de índices, congela essas camadas específicas da ResNet18
    if freeze_layers_idx:
        resnet_layers = list(rae.encoder.children())  # Obtém todas as camadas do encoder (ResNet18)
        for idx in freeze_layers_idx:
            if idx < len(resnet_layers):
                for param in resnet_layers[idx].parameters():
                    param.requires_grad = False  

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, rae.parameters()), lr=learning_rate)
    criterion = nn.MSELoss()
    # early_stopping = EarlyStopping(patience=80, enabled=eary_stopping_enabled, save_path=save_best_model_path)

    for epoch in range(num_epochs):
        total_loss = 0

        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)

            optimizer.zero_grad()
            reconstructed, mu, logvar = rae(x_batch)

            recon_loss = criterion(reconstructed, x_batch)
            kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_divergence

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}')
        
        # if early_stopping(val_acc, rae_classifier):
        #     print("⏹ Treinamento interrompido por Early Stopping!")
        #     break
    
    torch.save(rae.state_dict(), save_path)
    print(f"Model saved in {save_path}")
    print("✅ RAE Treinado!")

