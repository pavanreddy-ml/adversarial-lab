from abc import ABC, abstractmethod
from typing import Literal, List

from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.types import TensorType, TensorVariableType

class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    
    This class provides a framework for optimization algorithms used in training machine
    learning models. Subclasses should implement their own optimizer initialization.
    """
    def __init__(self) -> None:
        """
        Initialize the optimizer.
        
        Notes:
            - The actual optimizer is initialized in subclasses via `initialize_optimizer`.
        """
        self.optimizer = None

    @abstractmethod
    def initialize_optimizer(self):
        """
        Initialize the optimizer. Creates a new instance of the optimizer.
        """
        pass

    def apply(self,
              weights: List[TensorVariableType],
              gradients: List[TensorType]) -> None:
        """
        Apply gradients to update model weights.
        
        Parameters:
            weights (List[TensorVariableType]): The list of model parameters to be updated.
            gradients (List[TensorType]): The computed gradients corresponding to the weights.
        
        Notes:
            - If the optimizer has not been initialized, it will be initialized before applying gradients.
            - The `TensorOps.optimizers.apply` method is used to update the weights.
        """
        if self.optimizer is None:
            self.initialize_optimizer()
        self.tensor_ops.optimizers.apply(self.optimizer, weights, gradients)

    def has_param(self, param_name: str) -> bool:
        """
        Check if the optimizer has the given parameter.

        Parameters:
            param_name (str): The name of the parameter to check.

        Returns:
            bool: True if the parameter exists, False otherwise.
        """
        if not self.optimizer:
            raise ValueError("Optimizer has not been initialized.")
        return self.tensor_ops.optimizers.has_param(self.optimizer, param_name)

    def update_param(self, param_name: str, value) -> None:
        """
        Update the value of a parameter in the optimizer.

        Parameters:
            param_name (str): The name of the parameter to update.
            value: The new value to set.

        Raises:
            ValueError: If the parameter does not exist.
        """
        if not self.optimizer:
            raise ValueError("Optimizer has not been initialized.")
        self.tensor_ops.optimizers.update_param(self.optimizer, param_name, value)
        
    def set_framework(self, 
                      framework: Literal["tf", "torch"]
                      ) -> None:
        """
        Set the computational framework for the optimizer.
        
        Parameters:
            framework (Literal["tf", "torch"]): The framework to be set (TensorFlow or PyTorch).
        
        Raises:
            ValueError: If the provided framework is not 'tf' or 'torch'.
        
        Notes:
            - This method initializes `TensorOps` for the specified framework.
            - The optimizer is reinitialized to align with the new framework.
        """
        if framework not in ["tf", "torch"]:
            raise ValueError("framework must be either 'tf' or 'torch'")
        self.framework = framework
        self.tensor_ops = TensorOps(framework)
        self.initialize_optimizer()
