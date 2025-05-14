from abc import ABC, abstractmethod
from typing import Literal, List, Callable

from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.types import TensorType, TensorVariableType

class GradientEstimator(ABC):
    def __init__(self) -> None:
        self.optimizer = None

    @abstractmethod
    def calculate(self):
        pass
        
    def set_framework(self, 
                      framework: Literal["tf", "torch", "numpy"]
                      ) -> None:
        if framework not in ["numpy"]:
            raise ValueError("gradient estimator supports only 'numpy' framework")
        self.framework = framework
        self.tensor_ops = TensorOps(framework)

