from typing import Literal, List, Union
from .penalty_base import Penalty

import torch
import tensorflow as tf
from torch.nn import functional as F
from tensorflow.keras.losses import categorical_crossentropy


class LpNorm(Penalty):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 p: int = 2,
                 lambda_val: float = 1.0
                 ) -> None:
        super().__init__(framework)
        self.p = p
        self.lambda_val = lambda_val

    def calculate(self, 
                  predictions: Union[torch.Tensor, tf.Tensor], 
                  targets: Union[torch.Tensor, tf.Tensor]):
        return super().calculate(predictions, targets)

    def torch_op(self, 
                 predictions: torch.Tensor, 
                 targets: torch.Tensor
                 ) -> torch.Tensor:
        raise NotImplementedError("This method is not implemented for this class.")

    def tf_op(self, 
              predictions: tf.Tensor, 
              targets: tf.Tensor
              ) -> tf.Tensor:
        lp_norm = tf.reduce_sum(tf.abs(predictions) ** self.p) ** (1 / self.p)
        return lp_norm
