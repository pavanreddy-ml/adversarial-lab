from typing import Literal, List, Union, TypeVar, Generic, Dict, Any, Tuple, Union

import torch
import warnings
import numpy as np
import tensorflow as tf
from torch.nn import functional as F

from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.optimizers import Optimizer
from adversarial_lab.exceptions import IncompatibilityError
from .noise_generator_base import NoiseGenerator, NoiseGeneratorMeta
from adversarial_lab.core.constraints import PostOptimizationConstraint

class AdditiveNoiseGeneratorTF(NoiseGenerator):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 scale: List[int] = (-1, 1),
                 dist: Literal["normal", "uniform"] = "normal",
                 bounds: List[List[int]] = None,
                 bounds_type: Literal["relative", "absolute"] = "relative",
                 custom_mask: Union[np.ndarray, tf.Tensor] = None,
                 constraints: List[PostOptimizationConstraint] = None,
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(framework=framework,
                         bounds=bounds,
                         bounds_type=bounds_type,
                         custom_mask=custom_mask,
                         constraints=constraints)

        self.dist = dist
        self.scale = scale

    def generate_noise_meta(self,
                               sample: Union[np.ndarray, torch.Tensor, tf.Tensor],
                               bounds: List[List[int]] = None
                               ) -> tf.Variable:
        super().generate_noise_meta(sample, bounds)
    
        shape = sample.shape
        if shape[0] is None or shape[0] == 1:
            shape = shape[1:]

        if self.dist == "normal":
            noise_meta = tf.random.normal(shape)
            noise_meta = self.scale[0] + (self.scale[1] - self.scale[0]) * (noise_meta - tf.reduce_min(noise_meta)) / (tf.reduce_max(noise_meta) - tf.reduce_min(noise_meta))
        elif self.dist == "uniform":
            noise_meta = tf.random.uniform(shape, minval=self.scale[0], maxval=self.scale[1])
        else:
            raise ValueError(f"Unsupported distribution: {self.dist}")

        noise_meta = tf.Variable(noise_meta)

        noise_meta.assign(noise_meta * self._mask)
        self.apply_constraints(noise_meta)

        return [noise_meta]
    
    def get_noise(self,
                  noise_meta: List[tf.Variable]
                  ) -> np.ndarray:
        return noise_meta[0].numpy()

    def apply_noise(self,
                    sample: tf.Tensor | tf.Variable,
                    noise_meta: List[tf.Variable]
                    ) -> tf.Tensor:
        return sample + (self._mask * noise_meta[0])



class AdditiveNoiseGeneratorTorch(NoiseGenerator):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 use_constraints: bool,
                 epsilon: float,
                 scale: List[int] = None,
                 dist: Literal["normal", "uniform"] = "normal",
                 strict: bool = True,
                 *args,
                 **kwargs
                 ) -> None:
        self.tensor_ops = TensorOps(framework)

        self._mask = None
        self._strict = strict

    def generate(self,
                 shape: List[int]
                 ) -> torch.Tensor:
        raise NotImplementedError("generate Not Implemented for Torch")

    def apply_noise(self, *args, **kwargs):
        raise NotImplementedError("Apply Noise Not Implemented for Torch")

    def apply_constraints(self, *args, **kwargs):
        raise NotImplementedError("Apply Constrains Not Implemented for Torch")


class AdditiveNoiseGenerator(NoiseGenerator, metaclass=NoiseGeneratorMeta):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 scale: List[int] = (-1, 1),
                 dist: Literal["normal", "uniform"] = "normal",
                 bounds: List[List[int]] = None,
                 bounds_type: Literal["relative", "absolute"] = "relative",
                 custom_mask: Union[np.ndarray, tf.Tensor] = None,
                 constraints: List[PostOptimizationConstraint] = None,
                 *args,
                 **kwargs
                 ) -> None:
        self.framework: Literal["torch", "tf"] = framework
