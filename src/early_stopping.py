import torch

class EarlyStopping:
    """Para o treinamento se a acurácia de validação não melhorar após um número de épocas."""

    def __init__(self, patience=5, save_path="best_model.pth", enabled=True):
        """
        patience: Número de épocas sem melhora antes de interromper o treinamento.
        save_path: Caminho para salvar o melhor modelo.
        enabled: Se False, o Early Stopping não será aplicado.
        """
        self.patience = patience
        self.best_val_acc = 0.0  # Inicializa a melhor acurácia como 0
        self.counter = 0
        self.save_path = save_path
        self.enabled = enabled  # Define se o Early Stopping estará ativo

    def __call__(self, val_acc, model):
        """
        Chamada a cada época.
        """
        if not self.enabled:
            return False  # Se desativado, nunca interrompe o treinamento

        if val_acc > self.best_val_acc:
            print(f"🔹 Nova melhor acurácia! Salvando modelo... (Acurácia: {val_acc:.4f})")
            self.best_val_acc = val_acc
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)  # Salva o melhor modelo
        else:
            self.counter += 1
            print(f"🔸 EarlyStopping: {self.counter}/{self.patience} épocas sem melhora.")

        return self.counter >= self.patience  # Retorna True se o limite de paciência foi atingido
