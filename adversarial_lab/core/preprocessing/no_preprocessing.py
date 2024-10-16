from . import Preprocessing


class NoPreprocessing(Preprocessing):
    def __init__(self) -> None:
        pass

    def preprocess(self, input):
        return input