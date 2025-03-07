from typing import List

from . import Loss
from adversarial_lab.core.penalties import Penalty
from adversarial_lab.core.types import TensorType, LossType


class BinaryCrossEntropy(Loss):
    def __init__(self,
                 penalties: List[Penalty] = None,
                 from_logits: bool = False
                 ) -> None:
        super().__init__(penalties)
        self.from_logits = from_logits

    def calculate(self, 
                  target: TensorType,
                  predictions: TensorType,
                  logits: TensorType,
                  ) -> LossType:
        loss = self.tensor_ops.losses.binary_crossentropy(target=target, 
                                                          predictions=predictions, 
                                                          logits=logits, 
                                                          from_logits=self.from_logits)
        self.set_value(loss)
        return loss