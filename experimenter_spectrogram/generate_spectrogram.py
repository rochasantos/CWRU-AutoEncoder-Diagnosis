import matplotlib.pyplot as plt
from .spectrogram_generator import SpectrogramGenerator

def generate_spectrogram(signal, output_path):
    # Simulação de um sinal senoidal com ruído
    fs = 42000  # 48 kHz

    # Criando o gerador de espectrogramas
    spectrogram_generator = SpectrogramGenerator(window="hann", nperseg=200, noverlap=192, nfft=1600)

    # Gerando o espectrograma
    output = spectrogram_generator.generate(signal, sample_rate=fs, label="Falha_Bearing")

    # Exibir o espectrograma
    plt.imshow(output["spectrogram"])
    plt.axis("off")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100, format='png')
    plt.imsave(output_path, output["spectrogram"], cmap="jet")
    plt.close()