from typing import Literal, List, Union
from . import Loss
from .penalty.penalty_base import Penalty

import torch
import tensorflow as tf
from torch.nn import functional as F
from tensorflow.keras.losses import binary_crossentropy


class BinaryCrossEntropy(Loss):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 penalties: List[Penalty] = []
                 ) -> None:
        super().__init__(framework, penalties)

    def calculate(self, 
                  predictions: Union[torch.Tensor, tf.Tensor], 
                  targets: Union[torch.Tensor, tf.Tensor]):
        return super().calculate(predictions, targets)

    def torch_op(self, 
                 predictions: torch.Tensor, 
                 targets: torch.Tensor
                 ) -> torch.Tensor:
        loss = F.binary_cross_entropy(predictions, targets)
        self.set_value(loss)
        return loss

    def tf_op(self, 
              predictions: tf.Tensor, 
              targets: tf.Tensor
              ) -> tf.Tensor:
        loss = binary_crossentropy(targets, predictions, from_logits = True)
        loss = tf.reduce_mean(loss)
        self.set_value(loss)
        return loss
