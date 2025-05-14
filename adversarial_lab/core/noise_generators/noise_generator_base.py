from abc import ABC, abstractmethod, ABCMeta

import numpy as np

from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.optimizers import Optimizer

from typing import Literal, Union, List, Tuple, Optional, Callable
from adversarial_lab.core.types import TensorType, TensorVariableType, OptimizerType


class NoiseGenerator(ABC):
    def __init__(self,
                 mask: Optional[np.ndarray] = None,
                 requires_jacobian: bool = False) -> None:
        self._mask = mask
        self.requires_jacobian = requires_jacobian

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
    def construct_perturbation(self,
                               noise_meta: Union[TensorVariableType],
                               *args,
                               **kwargs
                               ) -> Union[TensorVariableType, TensorType]:
        pass

    def update(self,
               noise_meta: TensorVariableType,
               optimizer: OptimizerType | Optimizer,
               grads: TensorType,
               jacobian: TensorType = None,
               predictions: TensorType = None,
               target_vector: TensorType = None,
               true_class: int = None,
               predict_fn: Callable = None,
               *args,
               **kwargs
               ) -> None:
        optimizer.update(weights=noise_meta, gradients=grads)

    def get_mask(self) -> np.ndarray:
        return self.tensor_ops.numpy(self._mask) if self._mask is not None else None

    def set_framework(self,
                      framework: Literal["tf", "torch", "numpy"]
                      ) -> None:
        if framework not in ["tf", "torch", "numpy"]:
            raise ValueError("framework must be either 'tf' or 'torch'")
        self.framework = framework
        self.tensor_ops = TensorOps(framework)

        self._mask = self.tensor_ops.tensor(
            self._mask) if self._mask is not None else None
