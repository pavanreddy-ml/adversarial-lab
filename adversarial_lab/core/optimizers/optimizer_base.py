from abc import ABC, abstractmethod
from typing import Literal

from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.types import TensorType, TensorVariableType

class Optimizer:
    def __init__(self, framework: Literal["torch", "tf"]) -> None:
        self.optimizer = None
        self.framework = framework

    @abstractmethod
    def initialize_optimizer(self):
        pass

    @abstractmethod
    def apply(self,
              weights: TensorVariableType,
              gradients: TensorType | TensorVariableType):
        if self.optimizer is None:
            self.initialize_optimizer()
        self.tensor_ops.optimizers.apply(self.optimizer, weights, gradients)
        
    def set_framework(self, 
                      framework: Literal["tf", "torch"]
                      ) -> None:
        if framework not in ["tf", "torch"]:
            raise ValueError("framework must be either 'tf' or 'torch'")
        self.framework = framework
        self.tensor_ops = TensorOps(framework)
        self.initialize_optimizer()
