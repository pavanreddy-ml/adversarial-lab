import inspect

from . import PostOptimizationConstraint

from typing import Callable

class POConstraintFromFunction:
    """
    Create an instance of `PostOptimizationConstraint` from a function.

    This class provides a mechanism to convert a user-defined function into a 
    `PostOptimizationConstraint`, ensuring it follows the expected signature.
    """
    @staticmethod
    def create(function: Callable):
        """
        Create an instance of `PostOptimizationConstraint` from a function.

        The provided function must have the following signature:
        
        ```python
        def function(noise, *args, **kwargs):
        ```

        Parameters:
            function (Callable): The function to convert into a `PostOptimizationConstraint`.

        Returns:
            PostOptimizationConstraint: An instance of `PostOptimizationConstraint` 
            that applies the provided function.

        Raises:
            TypeError: If the function does not have the required parameters (`noise`, `*args`, `**kwargs`).

        Notes:
            - The function must accept at least three parameters: 
              `noise`, `*args`, and `**kwargs`.
            - If the function does not follow this signature, a `TypeError` is raised.
            - The created constraint will call the provided function when `apply()` is invoked.
        """
        sig = inspect.signature(function)
        params = sig.parameters
        
        has_noise = 'noise' in params
        has_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        if not has_noise:
            raise TypeError("Function to create POConstraint must have parameter: 'noise'.")

        if not has_args:
            raise TypeError("Function to create POConstraint must have parameter: '*args'.")
        
        if not has_kwargs:
            raise TypeError("Function to create POConstraint must have parameter: '**kwargs'.")

        class CustomPOConstraintFromFunction(PostOptimizationConstraint):
            """Custom constraint applying the user-provided function."""
            def __init__(self, func: Callable):
                """
                Initialize the constraint with the given function.

                Parameters:
                    func (Callable): The function to apply as a constraint.
                """
                self.function = func

            def apply(self, noise):
                return self.function(noise=noise)

        return CustomPOConstraintFromFunction(function)
    
    def __call__(self, function: Callable, *args, **kwds):
        """
        Callable interface to create a `PostOptimizationConstraint` from a function.

        This allows the instance of `POConstraintFromFunction` to be used as a decorator.

        Parameters:
            function (Callable): The function to convert into a `PostOptimizationConstraint`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            PostOptimizationConstraint: An instance of `PostOptimizationConstraint` 
            that applies the provided function.
        """
        return self.create(function)
