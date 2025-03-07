from typing import Literal
from . import PostOptimizationConstraint

from adversarial_lab.core.types import TensorVariableType


class POLpNorm(PostOptimizationConstraint):
    def __init__(self, 
                 epsilon: float = -1.0,
                 l_norm: str = "2",
                 max_iter: int = 100,
                 ) -> None:
        self.epsilon = epsilon
        self.l_norm = l_norm
        self.max_iter = max_iter
    
    def apply(self, 
              noise: TensorVariableType
              ) -> None:
        def compute_lp_norm(tensor, p):
            return self.tensor_ops.reduce_sum(self.tensor_ops.abs(tensor) ** p) ** (1.0 / p)

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

            if self.tensor_ops.abs(norm - self.epsilon) < 1e-6:
                break

        noise.assign(noise * low)
