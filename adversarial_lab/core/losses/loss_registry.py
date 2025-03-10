from typing import overload, Type, Union
from . import CategoricalCrossEntropy, BinaryCrossEntropy, MeanAbsoluteError, MeanSquaredError, Loss, DummyLoss

class LossRegistry:
    __losses = {
        "cce": CategoricalCrossEntropy,
        "bce": BinaryCrossEntropy,
        "mae": MeanAbsoluteError,
        "mse": MeanSquaredError,
    }

    @classmethod
    def get(cls, loss: Union[str, Loss]) -> Union[Type[Loss], Loss]:
        """
        Retrieve a loss function by name or return a `Loss` instance.

        This method retrieves the corresponding loss function class based on the provided
        string identifier. If `None` is provided, it returns the `DummyLoss` class.

        Parameters:
            loss (Union[str, Loss]): The name of the loss function as a string or an 
                instance of `Loss`.

        Returns:
            Union[Type[Loss], Loss]: The corresponding loss class or instance.

        Raises:
            TypeError: If `loss` is not a string, `None`, or an instance of `Loss`.
            ValueError: If the provided loss name does not exist in the registry.

        Notes:
            - Supported loss names: `"cce"` (Categorical Cross-Entropy), `"bce"` (Binary Cross-Entropy),
            `"mae"` (Mean Absolute Error), `"mse"` (Mean Squared Error).
            - If `loss` is `None`, the `DummyLoss` class is returned.
            - If the loss type is not found in the registry, a `ValueError` is raised.
        """
        if not isinstance(loss, (str, type(None))):
            raise TypeError(f"Invalid type for loss: {type(loss)}. Expected None, str or Loss.")
        
        if loss is None:
            return DummyLoss
        
        if loss not in cls.__losses:
            raise ValueError(f"Invalid value for loss: '{loss}'. Loss of this type does not exist.")
        return cls.__losses[loss]

