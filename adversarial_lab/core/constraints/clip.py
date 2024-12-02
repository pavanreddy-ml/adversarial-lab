from typing import Literal
from . import PostOptimizationConstraint

import tensorflow as tf
import torch


class POClip(PostOptimizationConstraint):
    def __init__(self, 
                 framework: Literal['torch'] | Literal['tf'],
                 min: float = -1.0,
                 max: float = 1.0
                 ) -> None:
        super().__init__(framework)
        self.min = min
        self.max = max

    def apply(self, 
              noise: torch.Tensor | tf.Variable, 
              ) -> torch.Tensor | tf.Tensor:
        return super().apply(noise)
    
    def torch_op(self, 
                 noise: torch.Tensor
                 ) -> None:
        pass
    
    def tf_op(self, 
              noise: tf.Variable
              ) -> None:
        noise.assign(tf.clip_by_value(noise, self.min, self.max))

