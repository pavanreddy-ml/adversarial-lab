from typing import Callable
from . import Preprocessing
import inspect

class PreprocessingFromFunction:
    @staticmethod
    def create(function: Callable):
        sig = inspect.signature(function)
        params = sig.parameters
        
        has_sample = 'sample' in params
        has_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        if not has_sample:
            raise TypeError("Function to create POConstraint must have parameter: 'sample'.")

        if not has_args:
            raise TypeError("Function to create POConstraint must have parameter: '*args'.")
        
        if not has_kwargs:
            raise TypeError("Function to create POConstraint must have parameter: '**kwargs'.")
        
        class CustomPreprocessingFromFunction(Preprocessing):
            def __init__(self, func: Callable):
                self.function = func

            def preprocess(self, input):
                return self.function(input)

        return CustomPreprocessingFromFunction(function)
    
    def __call__(self, function: Callable, *args, **kwds):
        return self.create(function)