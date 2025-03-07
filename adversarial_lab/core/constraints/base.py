from typing import Literal
from abc import ABC, abstractmethod

from adversarial_lab.core.types import TensorVariableType
from adversarial_lab.core.tensor_ops import TensorOps


class PostOptimizationConstraint(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def apply(self, 
              noise: TensorVariableType, 
              ) -> None:
        pass

    def set_framework(self, 
                      framework: Literal["tf", "torch"]
                      ) -> None:
        if framework not in ["tf", "torch"]:
            raise ValueError("framework must be either 'tf' or 'torch'")
        self.framework = framework
        self.tensor_ops = TensorOps(framework)