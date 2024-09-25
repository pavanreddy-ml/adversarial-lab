from typing import Literal
from . import Loss

import torch
import tensorflow as tf
from torch.nn import functional as F
from tensorflow.keras.losses import categorical_crossentropy


class CategoricalCrossEntropy(Loss):
    def __init__(self,
                 framework: Literal["torch", "tf"]
                 ) -> None:
        super().__init__(framework)

    def calculate(self, output, target):
        return super().calculate(output, target)

    def torch_op(self,
                 output: torch.Tensor,
                 target: torch.Tensor
                 ) -> torch.Tensor:
        return F.cross_entropy(output, target.argmax(dim=1))

    def tf_op(self,
              output: tf.Tensor,
              target: tf.Tensor
              ) -> tf.Tensor:
        return tf.reduce_mean(categorical_crossentropy(target, output))
