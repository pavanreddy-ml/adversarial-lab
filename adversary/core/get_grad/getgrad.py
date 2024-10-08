from . import GetGradsBase
from adversary.core.noise_generators import NoiseGenerator
from adversary.core.losses import Loss

from typing import Literal, List, Union

from abc import abstractmethod, ABC, ABCMeta

import torch
import tensorflow as tf


class GetGrads(GetGradsBase):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 loss: Loss) -> None:
        super().__init__(None, loss)
        self.framework = framework
        self.loss = loss

    def calculate(self,
                  model: Union[torch.nn.Module, tf.keras.Model],
                  sample: Union[torch.Tensor, tf.Tensor],
                  noise: Union[torch.Tensor, tf.Tensor],
                  noise_generator: NoiseGenerator,
                  targets: Union[torch.Tensor, tf.Tensor]):
        return super().calculate(model, sample, noise, noise_generator, targets)

    def torch_op(self,
                 model: Union[torch.nn.Module, tf.keras.Model],
                 sample: Union[torch.Tensor, tf.Tensor],
                 noise: Union[torch.Tensor, tf.Tensor],
                 noise_generator: NoiseGenerator,
                 targets: Union[torch.Tensor, tf.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError("Not implemented for Torch")

    def tf_op(self,
              model: tf.keras.Model,
              sample: tf.Tensor,
              noise: tf.Tensor,
              noise_generator: NoiseGenerator,
              targets: tf.Tensor
              ) -> List[tf.Tensor]:
        noise = tf.Variable(noise, trainable=True)

        with tf.GradientTape() as tape:
            tape.watch(noise)
            
            input = noise_generator.apply_noise(sample, noise)
            outputs = model(input)
            loss = self.loss.calculate(outputs, targets)
  
        gradients = tape.gradient(loss, noise)
        return gradients
