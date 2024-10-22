from typing import Literal, List, Union, TypeVar, Generic

import torch
import numpy as np
import tensorflow as tf
from torch.nn import functional as F

from . import NoiseGenerator, NoiseGeneratorMeta
from adversarial_lab.core.optimizers import Optimizer


class BoundedNoiseGeneratorTF(NoiseGenerator, metaclass=NoiseGeneratorMeta):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 use_constraints: bool,
                 epsilon: float,
                 scale: List[int] = None,
                 dist: Literal["normal", "uniform"] = "normal"
                 ) -> None:
        super().__init__(framework, use_constraints)

        if epsilon < 0 or epsilon > 1:
            raise ValueError("Epsilon must be between 0 and 1")

        self.epsilon = epsilon
        self.dist = dist

        if scale is None:
            self.scale = [-1, 1]
        else:
            self.scale = scale

        self.bounds = None

    def generate(self,
                 sample: Union[np.ndarray, torch.Tensor, tf.Tensor],
                 bounds: List[List[int]] = None
                 ) -> tf.Tensor:

        if not isinstance(sample, (np.ndarray, torch.Tensor, tf.Tensor)):
            raise TypeError(
                "Input must be of type np.ndarray, torch.Tensor, or tf.Tensor")

        shape = sample.shape
        if shape[0] is None:
            shape = shape[1:]

        if bounds is None:
            if len(shape) == 1:                             # For 1d Samples
                bounds = [[0, 0, shape[0], 1]]
            elif len(shape) >= 2:                            # For 2d Samples (Greyscale and 3 channel)
                bounds = [[0, 0, shape[0], shape[1]]]

        noise = []
        for bound in bounds:
            if len(bound) != 4:
                raise ValueError(
                    "Each bound should be a list of 4 values: [x, y, w, h]")
            x, y, w, h = bound
            x = max(0, x)
            y = max(0, y)
            w = min(w, shape[1] - x)
            h = min(h, shape[0] - y)

            if len(shape) == 1:  # 1D tabular data
                noise_shape = (w,)
            elif len(shape) == 2:  # 2D grayscale images
                noise_shape = (h, w)
            elif len(shape) == 3:  # 3D 3-channel images
                noise_shape = (h, w, shape[2])

            if self.dist == "normal":
                region_noise = tf.random.normal(noise_shape) * self.epsilon
            elif self.dist == "uniform":
                region_noise = tf.random.uniform(
                    noise_shape, minval=-1, maxval=1) * self.epsilon
            else:
                raise ValueError(f"Unsupported distribution: {self.dist}")

            current_min, current_max = tf.reduce_min(
                region_noise), tf.reduce_max(region_noise)
            target_min, target_max = self.scale
            region_noise = (region_noise - current_min) / (current_max -
                                                           current_min) * (target_max - target_min) + target_min

            noise.append(region_noise)

        self._set_bounds(bounds)
        return noise

    def apply_noise(self, sample: Union[tf.Tensor, tf.Variable], noise: List[tf.Variable]) -> tf.Tensor:
        if not isinstance(sample, tf.Variable):
            sample = tf.Variable(sample)
        
        sample_copy = tf.identity(sample)
        sample_copy = tf.Variable(sample_copy)

        for i, bound in enumerate(self.bounds):
            x, y, w, h = bound

            updated_region = sample_copy[y:y+h, x:x+w] + noise[i]
            sample_copy[y:y+h, x:x+w].assign(updated_region)

        return sample_copy

    def apply_constraints(self,
                          noise: List[tf.Variable]
                          ) -> None:
        min_value = self.scale[0] * self.epsilon
        max_value = self.scale[1] * self.epsilon

        for noise_tensor in noise:
            noise_tensor.assign(tf.clip_by_value(
                noise_tensor, clip_value_min=min_value, clip_value_max=max_value))

    def _set_bounds(self,
                    bounds: List[List[int]]
                    ) -> None:
        self.bounds = bounds

    def _validate_bounds(self, noise: List[tf.Tensor]) -> None:
        if self.bounds is None:
            raise ValueError(
                "Bounds are not set. Please call generate() before applying noise.")

        if len(noise) != len(self.bounds):
            raise ValueError(
                f"The number of noise tensors ({len(noise)}) does not match the number of bounds ({len(self.bounds)}).")

        for i, (bound, noise_tensor) in enumerate(zip(self.bounds, noise)):
            _, _, w, h = bound
            noise_shape = noise_tensor.shape

            if noise_shape[1] != w or noise_shape[0] != h:
                raise ValueError(
                    f"Shape of noise tensor at index {i} ({noise_shape}) does not match the corresponding bound dimensions ({w}, {h}).")


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
