from typing import Literal, List, Union, TypeVar, Generic

import torch
import numpy as np
import tensorflow as tf
from torch.nn import functional as F

from . import NoiseGenerator, NoiseGeneratorMeta
from .additive_noise_generator import AdditiveNoiseGeneratorTF, AdditiveNoiseGeneratorTorch
from adversarial_lab.core.optimizers import Optimizer
from adversarial_lab.exceptions import IncompatibilityError


class BoundedNoiseGeneratorTF(AdditiveNoiseGeneratorTF):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 use_constraints: bool,
                 epsilon: float,
                 scale: List[int] = None,
                 dist: Literal["normal", "uniform"] = "normal",
                 strict: bool = True
                 ) -> None:
        super().__init__(framework, use_constraints, epsilon, scale, dist)

        self._bounds = None
        self._strict = strict

    def generate(self,
                 sample: Union[np.ndarray, torch.Tensor, tf.Tensor],
                 bounds: List[List[int]] = None
                 ) -> tf.Tensor:

        if not isinstance(sample, (np.ndarray, torch.Tensor, tf.Tensor)):
            raise TypeError(
                "Input must be of type np.ndarray, torch.Tensor, or tf.Tensor")

        shape = sample.shape
        if shape[0] is None or shape[0] == 1:
            shape = shape[1:]

        if self._bounds is None and bounds is None:
            if len(shape) == 1:                             # For 1d Samples
                bounds = [[0, 0, shape[0], 1]]
            elif len(shape) >= 2:                            # For 2d Samples (Greyscale and 3 channel)
                bounds = [[0, 0, shape[0], shape[1]]]

            self.set_bounds(bounds)
    
        noise = [super(BoundedNoiseGeneratorTF, self).generate(sample)[0] for _ in range(len(self._bounds))]
        self.apply_constraints(noise)
        return noise

    def apply_noise(self, 
                    sample: Union[tf.Tensor, tf.Variable], 
                    noise: List[tf.Variable]) -> tf.Tensor:
        for i in range(len(noise)):
            sample = sample + noise[i]
        return sample

    def apply_constraints(self, 
                          noise: List[tf.Variable]
                          ) -> List[tf.Variable]:
        noise = super().apply_constraints(noise)
        for i, noise_tensor in enumerate(noise):
            shape = noise_tensor.shape
            if shape[0] == 1 or shape[0] == None:
                    shape = shape[1:]
            x, y, w, h = self._bounds[i]
            mask = np.zeros(shape)
            mask[y:y+h, x:x+w] = 1
            mask = tf.convert_to_tensor(mask, dtype=tf.float32)
            noise_tensor.assign(noise_tensor * mask)
        return noise

    def set_bounds(self,
                    bounds: List[List[int]]
                    ) -> None:
        if self._bounds is None:
            self._bounds = bounds
        elif not self._strict and self._bounds is not None:
            Warning("Bounds are already set. Overwriting bounds.")
            self._bounds = bounds
        elif self._strict and self._bounds is not None:
            raise AttributeError("Bounds are already set. Create a new instance of BoundedNoiseGenerator to set new bounds.")

    def get_bounds(self) -> List[List[int]]:
        return self._bounds

    def _validate_bounds(self, noise: List[tf.Tensor]) -> None:
        pass


class BoundedNoiseGeneratorTorch(NoiseGenerator):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 epsilon: float,
                 scale: List[int] = None,
                 dist: Literal["normal", "uniform"] = "normal"
                 ) -> None:
        self.epsilon = epsilon
        self.dist = dist

        self.scale = scale

    def generate(self,
                 shape: List[int]
                 ) -> torch.Tensor:
        raise NotImplementedError("generate Not Implemented for Torch")

    def apply_noise(self, *args, **kwargs):
        raise NotImplementedError("Apply Noise Not Implemented for Torch")

    def apply_constraints(self, *args, **kwargs):
        raise NotImplementedError("Apply Constrains Not Implemented for Torch")


T = TypeVar('T', BoundedNoiseGeneratorTF, BoundedNoiseGeneratorTorch)


class BoundedNoiseGenerator(Generic[T], metaclass=NoiseGeneratorMeta):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 use_constraints: bool,
                 epsilon: float,
                 scale: List[int] = None,
                 dist: Literal["normal", "uniform"] = "normal") -> None:
        self.framework: Literal["torch", "tf"] = framework
        self.use_constraints: bool = use_constraints
