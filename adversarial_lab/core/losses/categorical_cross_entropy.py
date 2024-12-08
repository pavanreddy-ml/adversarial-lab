from typing import Literal, List, Union
from . import Loss

import torch
import tensorflow as tf
from torch.nn import functional as F
from tensorflow.keras.losses import categorical_crossentropy

from . import Penalty


class CategoricalCrossEntropy(Loss):
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
        loss = F.cross_entropy(predictions, targets)
        self.set_value(loss)
        return loss

    def tf_op(self, 
              predictions: tf.Tensor, 
              targets: tf.Tensor
              ) -> tf.Tensor:
        loss = categorical_crossentropy(targets, predictions)
        loss = tf.reduce_mean(loss)
        self.set_value(loss)
        return loss
