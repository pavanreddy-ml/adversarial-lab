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
                 lambda_val: float = 1.0,
                 on: Literal["noise", "predictions"] = 'noise',
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(framework, *args, **kwargs)
        self.p = p
        self.lambda_val = lambda_val

        if on not in ["noise", "predictions"]:
            raise ValueError(f"Unsupported value for 'on': {on}")
        
        self.on = on

    def calculate(self, 
                  predictions: Union[torch.Tensor, tf.Tensor], 
                  noise: Union[torch.Tensor, tf.Tensor],
                  *args, 
                  **kwargs):
        return super().calculate(*args, **kwargs)

    def torch_op(self, 
                 predictions: torch.Tensor, 
                 noise: torch.Tensor,
                 *args,
                 **kwargs
                 ) -> torch.Tensor:
        raise NotImplementedError("This method is not implemented for this class.")

    def tf_op(self, 
              predictions: tf.Tensor, 
              noise: tf.Tensor | tf.Variable,
              *args,
              **kwargs
              ) -> tf.Tensor:
        if self.on == "noise":
            apply_on = noise
        elif self.on == "predictions":
            apply_on = predictions
            
        lp_norm = tf.reduce_sum(tf.abs(apply_on) ** self.p) ** (1 / self.p)
        self.set_value(lp_norm)
        return lp_norm

    def __repr__(self) -> str:
        return super().__repr__() + f" p: {self.p}, lambda: {self.lambda_val}, on: {self.on}"