from typing import Callable
from . import Preprocessing
import inspect

class PreprocessingFromFunction:
    @staticmethod
    def create(function: Callable):
        sig = inspect.signature(function)
        params = sig.parameters
        
        has_data_arg = 'data' in params
        has_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        if not has_data_arg:
            raise TypeError("Function to create POCoPreprocessingnstraint must have parameter: 'data'.")

        if not has_args:
            raise TypeError("Function to create Preprocessing must have parameter: '*args'.")
        
        if not has_kwargs:
            raise TypeError("Function to create Preprocessing must have parameter: '**kwargs'.")
        
        class CustomPreprocessingFromFunction(Preprocessing):
            def __init__(self, func: Callable):
                self.function = func

            def preprocess(self, data):
                return self.function(data)

        return CustomPreprocessingFromFunction(function)
    
    def __call__(self, function: Callable, *args, **kwds):
        return self.create(function)