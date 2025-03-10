from typing import Callable
from . import Loss
import inspect

from adversarial_lab.core.types import TensorType

class LossFromFunction:
    """
    Create an instance of `Loss` from a function.

    This class provides a mechanism to convert a user-defined function into a 
    `Loss` instance, ensuring it follows the expected signature.
    """    
    @staticmethod
    def create(function: Callable):
        """
        Create an instance of `Loss` from a function.

        The provided function must have the following signature:
        
        ```python
        def function(target, predictions, logits, from_logits, *args, **kwargs):
        ```

        Parameters:
            function (Callable): The function to convert into a `Loss`.

        Returns:
            Loss: An instance of `Loss` that applies the provided function.

        Raises:
            TypeError: If the function does not have the required parameters 
            (`target`, `predictions`, `logits`, `from_logits`, `*args`, `**kwargs`).

        Notes:
            - The function must accept at least six parameters:
              `target`, `predictions`, `logits`, `from_logits`, `*args`, and `**kwargs`.
            - If the function does not follow this signature, a `TypeError` is raised.
            - The created loss function will call the provided function when `calculate()` is invoked.
        """
        sig = inspect.signature(function)
        params = sig.parameters

        has_preds = 'predictions' in params
        has_target = 'target' in params
        has_logits = 'logits' in params
        has_from_logits = 'from_logits' in params
        has_args = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        if not has_preds:
            raise TypeError(
                "Function to create Loss must have parameter: 'predictions'.")

        if not has_target:
            raise TypeError(
                "Function to create Loss must have parameter: 'target'.")

        if not has_logits:
            raise TypeError(
                "Function to create Loss must have parameter: 'logits'.")

        if not has_from_logits:
            raise TypeError(
                "Function to create Loss must have parameter: 'from_logits'.")

        if not has_args:
            raise TypeError(
                "Function to create Loss must have parameter: '*args'.")

        if not has_kwargs:
            raise TypeError(
                "Function to create Loss must have parameter: '**kwargs'.")

        class CustomLossFromFunction(Loss):
            """Custom loss function applying the user-provided function."""
            def __init__(self, func: Callable):
                self.function = func

            def calculate(self,
                          target: TensorType,
                          predictions: TensorType,
                          logits: TensorType,
                          from_logits: bool = False,
                          *args,
                          **kwargs):
                return self.function(target, predictions, logits, from_logits, *args, **kwargs)

        return CustomLossFromFunction(function)

    def __call__(self, function: Callable, *args, **kwds):
        """
        Callable interface to create a `Loss` from a function.

        This allows an instance of `LossFromFunction` to be used as a decorator.

        Parameters:
            function (Callable): The function to convert into a `Loss`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Loss: An instance of `Loss` that applies the provided function.
        """
        return self.create(function)
