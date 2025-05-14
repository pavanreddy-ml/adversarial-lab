from abc import ABC, abstractmethod

from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.types import TensorVariableType

from typing import Literal


class PostOptimizationConstraint(ABC):
    """
    Base class for post-optimization constraints.
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def apply(self, 
              noise: TensorVariableType, 
              *args,
              **kwargs
              ) -> None:
        """
        Apply the constraint to the noise inplace.

        Parameters:
            noise (TensorVariableType): The noise to apply the constraint to.
        """
        pass

    def set_framework(self, 
                      framework: Literal["tf", "torch", "numpy"]
                      ) -> None:
        """
        Set the framework for the constraint.

        Parameters:
            framework (str): The framework to set, tf or torch.

        Raises:
            ValueError: If the framework is not 'tf' or 'torch'.
        """
        if framework not in ["tf", "torch", "numpy"]:
             raise ValueError("framework must be either 'tf', 'torch' or 'numpy'")
        self.framework = framework
        self.tensor_ops = TensorOps(framework)