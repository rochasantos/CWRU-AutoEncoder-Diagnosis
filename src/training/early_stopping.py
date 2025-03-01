import torch

class EarlyStopping:
    """Stop training if validation accuracy does not improve after a number of epochs."""

    def __init__(self, patience=10, save_path="best_model.pth", enabled=True):
        """
        patience: Number of epochs without improvement before stopping training.
        save_path: Path to save the best model.
        enabled: If False, Early Stopping will not be applied.
        """
        self.patience = patience
        self.best_val_acc = 0.0  # Inicializa a melhor acurácia como 0
        self.counter = 0
        self.save_path = save_path
        self.enabled = enabled  # Define se o Early Stopping estará ativo

    def __call__(self, val_acc, model):
        """
        Called every season.
        """
        if not self.enabled:
            return False  # Se desativado, nunca interrompe o treinamento

        if val_acc > self.best_val_acc:
            print(f"🔹 New better accuracy! Saving model... (Accuracy: {val_acc:.4f})")
            self.best_val_acc = val_acc
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)  # Salva o melhor modelo
        else:
            self.counter += 1
            print(f"🔸 EarlyStopping: {self.counter}/{self.patience} times without improvement.")

        return self.counter >= self.patience  # Retorna True se o limite de paciência foi atingido
