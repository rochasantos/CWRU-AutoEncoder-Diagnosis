import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_rae(rae, input_dataset, target_dataset, num_epochs=50, batch_size=32, learning_rate=1e-3, device="cuda"):
    """
    Treina o RAE comparando a reconstrução com outra amostra (target) em vez da entrada original.

    Args:
        rae: Modelo RAE a ser treinado.
        input_dataset: Dataset de entrada (espectrogramas de entrada).
        target_dataset: Dataset de saída (espectrogramas esperados para reconstrução).
        num_epochs: Número de épocas de treinamento.
        batch_size: Tamanho do batch.
        learning_rate: Taxa de aprendizado.
        device: Dispositivo de treinamento ("cuda" ou "cpu").
    """
    rae.to(device)
    rae.train()

    input_loader = DataLoader(input_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(rae.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0

        for (x_input, _), (x_target, _) in zip(input_loader, target_loader):
            x_input, x_target = x_input.to(device), x_target.to(device)

            optimizer.zero_grad()

            # 🔹 Passa pelo encoder e decoder
            reconstructed, mu, logvar = rae(x_input)

            # 🔹 Calcula a perda com x_target em vez de x_input
            recon_loss = criterion(reconstructed, x_target)
            kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_loss + kl_divergence
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(input_loader):.4f}')

    torch.save(rae.state_dict(), "rae_spectro_model.pth")
    print("✅ RAE Treinado com comparação alternativa!")
