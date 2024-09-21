from abc import ABC, abstractmethod

from typing import Union, Any

import numpy as np
from torch.nn import Module as TorchModel
from tensorflow.keras.models import Model as TFModel

class WhiteBoxAttack(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def attack(self, 
               model: Union[TorchModel, TFModel],
               sample: Any,
               confidence_threshold: float,
               *args,
               **kwargs
               ) -> np.ndarray:
        pass