from abc import ABC, abstractmethod, ABCMeta
from typing import Literal, Union, List, Tuple
import numpy as np

from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.optimizers import Optimizer
from adversarial_lab.core.types import TensorType, TensorVariableType, OptimizerType


class NoiseGenerator(ABC):   
    def __init__(self, mask: np.ndarray = None) -> None:
        self._mask = mask
    
    @abstractmethod
    def generate_noise_meta(self, 
                            sample: TensorType | np.ndarray, 
                            *args, 
                            **kwargs
                            ) -> List[TensorVariableType]:
        pass

    @abstractmethod
    def get_noise(self, 
                  noise_meta: Union[TensorVariableType]
                  ) -> np.ndarray:
        pass

    @abstractmethod
    def apply_noise(self, 
                    sample: TensorType,
                    noise_meta: Union[TensorVariableType], 
                    *args, 
                    **kwargs
                    ) -> Union[TensorVariableType, TensorType]:
        pass

    def apply_gradients(self, 
                        noise_meta: TensorVariableType, 
                        grads: TensorType, 
                        logit_grads: TensorType, 
                        optimizer: OptimizerType | Optimizer) -> None:
        optimizer.apply(weights=noise_meta, gradients=grads)

    def get_mask(self) -> np.ndarray:
        return self._mask

    def set_framework(self, 
                      framework: Literal["tf", "torch"]
                      ) -> None:
        if framework not in ["tf", "torch"]:
            raise ValueError("framework must be either 'tf' or 'torch'")
        self.framework = framework
        self.tensor_ops = TensorOps(framework)
