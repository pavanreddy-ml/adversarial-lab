from abc import ABC, abstractmethod
from typing import Literal

import warnings
import traceback

from adversarial_lab.core.types import LossType, TensorType
from adversarial_lab.core.tensor_ops import TensorOps

class Penalty(ABC):
    def __init__(self, 
                 *args,
                 **kwargs) -> None:
        self.value = None
        self.warned = False

    @abstractmethod
    def calculate(self, 
                  noise: TensorType,
                  *args, 
                  **kwargs):
        pass
    
    def set_value(self, 
                  value: LossType
                  ) -> None:
        if value is None:
            self.value = None
            return
        
        try:
            if self.framework == "torch":
                self.value = value.item()
            elif self.framework == "tf":
                self.value = float(value.numpy())
        except Exception as e:
            if not self.warned:
                self.warned = True
                warnings.warn(f"Error while setting value: {e}. Traceback: {traceback.format_exc()}")

            self.value = None

    def has_param(self, param_name: str) -> bool:
        """
        Check if the penalty instance has the given parameter.

        Args:
            param_name (str): The name of the parameter to check.

        Returns:
            bool: True if the parameter exists, False otherwise.
        """
        return hasattr(self, param_name)

    def update_param(self, param_name: str, value) -> None:
        """
        Update the value of a parameter in the penalty instance.

        Args:
            param_name (str): The name of the parameter to update.
            value: The new value to set.

        Raises:
            ValueError: If the parameter does not exist.
        """
        if not hasattr(self, param_name):
            raise ValueError(f"Unknown parameter: {param_name}")
        setattr(self, param_name, value)

    def set_framework(self, 
                      framework: Literal["tf", "torch", "numpy"]
                      ) -> None:
        if framework not in ["tf", "torch", "numpy"]:
            raise ValueError("framework must be either 'tf', 'torch' or 'numpy'")
        self.framework = framework
        self.tensor_ops = TensorOps(framework)

    def __repr__(self) -> str:
        return self.__class__.__name__
