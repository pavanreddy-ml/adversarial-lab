from typing import Literal
from . import PostOptimizationConstraint

from adversarial_lab.core.types import TensorVariableType

class POClip(PostOptimizationConstraint):
    def __init__(self, 
                 framework: Literal["torch", "tf"],
                 min: float = -1.0,
                 max: float = 1.0
                 ) -> None:
        super().__init__(framework)
        self.min = min
        self.max = max

    def apply(self, 
              noise: TensorVariableType, 
              ) -> None:
        clipped_value = self.tensor_ops.clip(noise, self.min, self.max)
        self.tensor_ops.assign(noise, clipped_value)

