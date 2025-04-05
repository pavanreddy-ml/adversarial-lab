from . import Callback


class ChangeParamsOnConvergence(Callback):
    def __init__(self, model, params):
        self.params = params

    def on_epoch_end(self, epoch, logs=None):
        pass