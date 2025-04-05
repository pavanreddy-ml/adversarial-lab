from abc import ABC, abstractmethod


class Callback(ABC):
    def __init__(self):
        pass

    def on_epoch_end(self):
        pass

    def on_batch_end(self):
        pass

