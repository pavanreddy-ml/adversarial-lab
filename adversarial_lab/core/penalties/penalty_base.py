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

    def set_framework(self, 
                      framework: Literal["tf", "torch"]
                      ) -> None:
        if framework not in ["tf", "torch"]:
            raise ValueError("framework must be either 'tf' or 'torch'")
        self.framework = framework
        self.tensor_ops = TensorOps(framework)

    def __repr__(self) -> str:
        return self.__class__.__name__
