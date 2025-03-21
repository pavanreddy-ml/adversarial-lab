from typing import List

from . import Loss
from adversarial_lab.core.penalties import Penalty
from adversarial_lab.core.types import TensorType, LossType


class MeanAbsoluteError(Loss):
    """
    Compute the mean absolute error loss.
    """

    def __init__(self,
                 penalties: List[Penalty] = None,
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(penalties)

    def calculate(self,
                  target: TensorType,
                  predictions: TensorType,
                  logits: TensorType,
                  from_logits: bool = False
                  ) -> LossType:
        loss = self.tensor_ops.losses.mean_absolute_error(target=target,
                                                          predictions=predictions)
        self.set_value(loss)
        return loss
