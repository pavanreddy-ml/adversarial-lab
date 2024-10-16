from typing import Literal, List, Union
from . import Loss

import torch
import tensorflow as tf
from torch.nn import functional as F
from tensorflow.keras.losses import binary_crossentropy


class BinaryCrossEntropy(Loss):
    def __init__(self,
                 framework: Literal["torch", "tf"]
                 ) -> None:
        super().__init__(framework)

    def calculate(self, 
                  predictions: Union[torch.Tensor, tf.Tensor], 
                  targets: Union[torch.Tensor, tf.Tensor]):
        return super().calculate(predictions, targets)

    def torch_op(self, 
                 predictions: torch.Tensor, 
                 targets: torch.Tensor
                 ) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)
        loss = F.binary_cross_entropy(predictions, targets)
        return loss

    def tf_op(self, 
              predictions: tf.Tensor, 
              targets: tf.Tensor
              ) -> tf.Tensor:
        loss = binary_crossentropy(targets, predictions, from_logits = True)
        return tf.reduce_mean(loss)