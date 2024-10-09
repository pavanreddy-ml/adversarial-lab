from typing import Callable
from . import Preprocessing

class PreprocessingFromFunction:
    @staticmethod
    def create(function: Callable):
        class CustomPreprocessingFromFunction(Preprocessing):
            def __init__(self, func: Callable):
                self.function = func

            def preprocess(self, input):
                return self.function(self, input)

        return CustomPreprocessingFromFunction(function)