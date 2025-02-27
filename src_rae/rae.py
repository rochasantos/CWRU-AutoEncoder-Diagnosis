import torch
import torch.nn as nn

class RAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=2):
        super(RAE, self).__init__()

        # 🔹 Encoder (LSTM)
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Média do espaço latente
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log da variância

        # 🔹 Decoder (LSTM)
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)  # Projeta para o espaço oculto do LSTM
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Desvio padrão
        eps = torch.randn_like(std)  # Ruído gaussiano
        return mu + eps * std  # Amostragem do espaço latente

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape

        # 🔹 Passa pelo encoder LSTM
        _, (h_n, _) = self.encoder(x)
        encoded = h_n[-1]  # Usa apenas a última saída do LSTM

        # 🔹 Calcula média e variância para o espaço latente
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)

        # 🔹 Decodificação
        hidden = self.fc_dec(z).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        decoded, _ = self.decoder(hidden)

        return decoded, mu, logvar
