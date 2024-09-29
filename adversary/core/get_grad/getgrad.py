from . import GetGradsBase

from typing import Literal, List, Union

from abc import abstractmethod, ABC, ABCMeta

import torch
import tensorflow as tf



class GetGrads(GetGradsBase):
    def __init__(self, 
                 framework: Literal["torch", "tf"],
                 loss) -> None:
        super().__init__(None, loss)
        self.framework = framework
        self.loss = loss

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