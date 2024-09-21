from typing import overload, Type, Union
from . import CategoricalCrossEntropy, Loss

class LossRegistry:
    __losses = {
        "cce": CategoricalCrossEntropy
    }

    @classmethod
    def get(cls, loss: Union[str, Loss]) -> Union[Type[Loss], Loss]:
        if loss not in cls.__losses:
            raise ValueError(f"Invalid value for loss: '{loss}'. Loss of this type does not exist.")
        return cls.__losses[loss]
