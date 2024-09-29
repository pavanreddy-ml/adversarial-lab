from typing import Literal, List, Union
from . import NoiseGenerator

import torch
import tensorflow as tf
from torch.nn import functional as F


class AdditiveNoiseGenerator(NoiseGenerator):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 epsilon: float,
                 scale: List[int] = None,
                 dist: Literal["normal", "uniform"] = "normal"
                 ) -> None:
        super().__init__(framework)
        self.epsilon = epsilon
        self.dist = dist

        self.scale = scale

    def generate(self, 
                 shape: List[int]):
        return super().generate(shape)

    def torch_op(self, 
                 shape: List[int]
                 ) -> torch.Tensor:
        if self.dist == "normal":
            noise = torch.randn(shape, device='cuda' if torch.cuda.is_available() else 'cpu')
            noise = torch.clamp(noise, -1, 1) * self.epsilon
        elif self.dist == "uniform":
            noise = torch.rand(shape, device='cuda' if torch.cuda.is_available() else 'cpu') * 2 - 1
            noise *= self.epsilon
        else:
            raise ValueError(f"Unsupported distribution: {self.dist}")
        
        if self.scale is not None:
            current_min, current_max = noise.min(), noise.max()
            target_min, target_max = self.scale
            noise = (noise - current_min) / (current_max - current_min) * (target_max - target_min) + target_min
        
        return noise

    def tf_op(self, 
              shape: List[int]) -> tf.Tensor:
        if self.dist == "normal":
            noise = tf.random.normal(shape)
            noise = tf.clip_by_value(noise, -1, 1) * self.epsilon
        elif self.dist == "uniform":
            noise = tf.random.uniform(shape, minval=-1, maxval=1) * self.epsilon
        else:
            raise ValueError(f"Unsupported distribution: {self.dist}")
        
        if self.scale is not None:
            current_min, current_max = tf.reduce_min(noise), tf.reduce_max(noise)
            target_min, target_max = self.scale
            noise = (noise - current_min) / (current_max - current_min) * (target_max - target_min) + target_min
        
        return noise