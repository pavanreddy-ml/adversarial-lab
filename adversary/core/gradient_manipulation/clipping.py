from typing import Literal, Union, List
import torch
import tensorflow as tf
from . import GradientManipulation

class GradientClipping(GradientManipulation):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 max: float = None,
                 min: float = None) -> None:
        super().__init__(framework)
        self.max = max
        self.min = min

    def apply(self, gradients: Union[torch.Tensor, tf.Tensor]) -> Union[torch.Tensor, tf.Tensor]:
        return super().apply(gradients)

    def torch_op(self, gradients: torch.Tensor) -> torch.Tensor:
        if self.max is not None and self.min is not None:
            return torch.clamp(gradients, min=self.min, max=self.max)
        elif self.max is not None:
            return torch.clamp(gradients, max=self.max)
        elif self.min is not None:
            return torch.clamp(gradients, min=self.min)
        return gradients

    def tf_op(self, gradients: tf.Tensor) -> tf.Tensor:
        if self.max is not None and self.min is not None:
            return tf.clip_by_value(gradients, clip_value_min=self.min, clip_value_max=self.max)
        elif self.max is not None:
            return tf.where(gradients > self.max, self.max, gradients)
        elif self.min is not None:
            return tf.where(gradients < self.min, self.min, gradients)
        return gradients