from typing import Literal
from . import PostOptimizationConstraint

import tensorflow as tf
import torch


class POLpNorm(PostOptimizationConstraint):
    def __init__(self, 
                 framework: Literal['torch'] | Literal['tf'],
                 epsilon: float = -1.0,
                 l_norm: str = "2",
                 max_iter: int = 100
                 ) -> None:
        super().__init__(framework)
        self.epsilon = epsilon
        self.l_norm = l_norm
        self.max_iter = max_iter

    def apply(self, 
              noise: torch.Tensor | tf.Variable, 
              ) -> torch.Tensor | tf.Tensor:
        return super().apply(noise)
    
    def torch_op(self, 
                 noise: torch.Tensor
                 ) -> None:
        raise NotImplementedError("LpNorm constraint is not implemented for PyTorch yet.")
    
    def tf_op(self, 
              noise: tf.Variable
              ) -> None:
        def compute_lp_norm(tensor, p):
            return tf.reduce_sum(tf.abs(tensor) ** p) ** (1.0 / p)

        p = float(self.l_norm)
        low, high = 0.0, 1.0

        for _ in range(self.max_iter):
            scale_factor = (low + high) / 2.0
            scaled_noise = noise * scale_factor
            norm = compute_lp_norm(scaled_noise, p)

            if norm > self.epsilon:
                high = scale_factor
            else:
                low = scale_factor

            if tf.abs(norm - self.epsilon) < 1e-6:
                break

        noise.assign(noise * low)
