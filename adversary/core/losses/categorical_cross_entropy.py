from typing import Literal, List, Union
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

    def calculate(self, 
                  model: Union[torch.nn.Module, tf.keras.Model], 
                  inputs: Union[torch.Tensor, tf.Tensor], 
                  targets: Union[torch.Tensor, tf.Tensor]):
        return super().calculate(model, inputs, targets)

    def torch_op(self, 
                 model: torch.nn.Module, 
                 inputs: torch.Tensor, 
                 targets: torch.Tensor
                 ) -> List[torch.Tensor]:
        model.zero_grad()
        outputs = model(inputs)
        loss = self.loss.calculate(outputs, targets)
        loss.backward()
        return [param.grad for param in model.parameters() if param.grad is not None]

    def tf_op(self, 
              model: tf.keras.Model, 
              inputs: tf.Tensor, 
              targets: tf.Tensor
              ) -> List[tf.Tensor]:
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss = self.loss.calculate(outputs, targets)
        return tape.gradient(loss, model.trainable_variables)
