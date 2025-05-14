import copy

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False,
                 restore_best_weights=True, start_threshold=0.01):
        """
        Stops training early if the validation loss doesn't improve after a given patience,
        but only starts monitoring once the loss reaches a certain threshold.

        Args:
            patience (int): Number of epochs with no improvement after which training stops.
            min_delta (float): Minimum change in the monitored loss to qualify as improvement.
            verbose (bool): Whether to print detailed logs.
            restore_best_weights (bool): Restore model weights from best epoch.
            start_threshold (float): Minimum loss required before monitoring starts.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.start_threshold = start_threshold

        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        self.monitoring_started = False

    def __call__(self, current_loss, model):
        """
        Call this at the end of each epoch with the current validation loss.

        Args:
            current_loss (float): The validation loss this epoch.
            model (torch.nn.Module): Model to optionally store best weights from.

        Returns:
            bool: True if early stopping should occur.
        """
        # Aguarda até que a loss fique abaixo do limiar
        if not self.monitoring_started:
            if current_loss <= self.start_threshold:
                self.monitoring_started = True
                if self.verbose:
                    print(f"EarlyStopping: Monitoring started (loss={current_loss:.4f} <= threshold={self.start_threshold})")
            else:
                if self.verbose:
                    print(f"EarlyStopping: Waiting to start monitoring (loss={current_loss:.4f} > threshold={self.start_threshold})")
                return False  # continua o treinamento normalmente

        # Agora o monitoramento está ativo
        if self.best_loss is None:
            self.best_loss = current_loss
            if self.restore_best_weights:
                self.best_model_state = copy.deepcopy(model.state_dict())
            return False

        if current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement ({self.counter}/{self.patience}) | Best: {self.best_loss:.4f} | Current: {current_loss:.4f}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("EarlyStopping: Stopping training.")
                if self.restore_best_weights and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
                    if self.verbose:
                        print("EarlyStopping: Restored best model weights.")
                return True
        else:
            if self.verbose:
                print(f"EarlyStopping: Improvement! {self.best_loss:.4f} → {current_loss:.4f}")
            self.best_loss = current_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = copy.deepcopy(model.state_dict())

        return False
