from . import Callback


class EarlyStopping(Callback):
    def __init__(self, patience: int = 5, threshold: float = 0.0, on=""):
        super().__init__()
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.counter = 0
        self.on = on
        self.early_stop = False

    def on_epoch_end(self):
        pass

    def on_batch_end(self):
        pass
