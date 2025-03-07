from typing import Literal, List, Union
from .penalty_base import Penalty

from adversarial_lab.core.types import LossType, TensorType


class LpNorm(Penalty):
    def __init__(self,
                 p: int = 2,
                 lambda_val: float = 1.0,
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(*args, **kwargs)
        self.p = p
        self.lambda_val = lambda_val

    def calculate(self, 
                  noise: TensorType,
                  *args, 
                  **kwargs):
        lp_norm = self.tensor_ops.reduce_sum(self.tensor_ops.abs(noise) ** self.p) ** (1 / self.p)
        self.set_value(lp_norm)
        return lp_norm  

    def __repr__(self) -> str:
        return super().__repr__() + f" p: {self.p}, lambda: {self.lambda_val}, on: {self.on}"