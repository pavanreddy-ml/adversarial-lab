from typing import Callable
from . import Loss
import inspect

class LossFromFunction:
    @staticmethod
    def create(function: Callable):
        sig = inspect.signature(function)
        params = sig.parameters
        
        has_preds = 'predictions' in params
        has_target = 'target' in params
        has_logits = 'logits' in params
        has_from_logits = 'from_logits' in params
        has_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        if not has_preds:
            raise TypeError("Function to create Loss must have parameter: 'predictions'.")
        
        if not has_target:
            raise TypeError("Function to create Loss must have parameter: 'target'.")
        
        if not has_logits:
            raise TypeError("Function to create Loss must have parameter: 'logits'.")
        
        if not has_from_logits:
            raise TypeError("Function to create Loss must have parameter: 'from_logits'.")

        if not has_args:
            raise TypeError("Function to create Loss must have parameter: '*args'.")
        
        if not has_kwargs:
            raise TypeError("Function to create Loss must have parameter: '**kwargs'.")
        
        class CustomLossFromFunction(Loss):
            def __init__(self, func: Callable):
                self.function = func

            def preprocess(self, input):
                return self.function(input)

        return CustomLossFromFunction(function)
    
    def __call__(self, function: Callable, *args, **kwds):
        return self.create(function)