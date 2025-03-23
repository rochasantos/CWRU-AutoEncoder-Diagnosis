import copy

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False, restore_best_weights=True):
        """
        Stops training early if the validation loss doesn't improve after a given patience.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored loss to qualify as improvement.
            verbose (bool): Whether to print messages when loss improves or not.
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights

        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, current_loss, model):
        """
        Call this method at the end of each epoch.

        Args:
            current_loss (float): Current validation loss.
            model (torch.nn.Module): The model being trained.

        Returns:
            bool: True if training should stop early, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = current_loss
            if self.restore_best_weights:
                self.best_model_state = copy.deepcopy(model.state_dict())
            return False

        if current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement in loss ({self.counter}/{self.patience})")
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
                print(f"EarlyStopping: Improvement in loss from {self.best_loss:.4f} to {current_loss:.4f}")
            self.best_loss = current_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = copy.deepcopy(model.state_dict())

        return False
