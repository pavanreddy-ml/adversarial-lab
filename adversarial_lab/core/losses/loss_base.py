from abc import ABC, abstractmethod
from typing import Literal, List

import warnings
import traceback

from adversarial_lab.core.penalties import Penalty
from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.types import LossType, TensorType



class Loss():
    def __init__(self, 
                 penalties: List[Penalty]
                 ) -> None:
        self.value = None
        penalties = penalties if penalties is not None else []

        self.warned = False

        if not all(isinstance(penalty, Penalty) for penalty in penalties):
            raise TypeError("penalties must be a list of Penalty instances")
        
        self.penalties = penalties

    @abstractmethod
    def calculate(self, 
                  target: TensorType, 
                  predictions: TensorType, 
                  logits: TensorType, 
                  *args, 
                  **kwargs
                  ) -> LossType:
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
