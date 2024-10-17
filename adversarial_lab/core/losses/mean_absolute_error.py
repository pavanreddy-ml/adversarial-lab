from typing import Literal, List, Union
from . import Loss

import torch
import tensorflow as tf
from torch.nn import functional as F
from tensorflow.keras.losses import MeanAbsoluteError as MAE_tf


class MeanAbsoluteError(Loss):
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
        loss = F.l1_loss(predictions, targets)
        return loss

    def tf_op(self, 
              predictions: tf.Tensor, 
              targets: tf.Tensor
              ) -> tf.Tensor:
        mae = MAE_tf()
        loss = mae(targets, predictions)
        return loss