from typing import Literal, List, Union, TypeVar, Generic

import torch
import warnings
import numpy as np
import tensorflow as tf
from torch.nn import functional as F

from . import NoiseGenerator, NoiseGeneratorMeta
from adversarial_lab.exceptions import IncompatibilityError
from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.optimizers import Optimizer
from .additive_noise_generator import AdditiveNoiseGeneratorTF, AdditiveNoiseGeneratorTorch


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

        self.tensor_ops = TensorOps(framework)

        self._mask = None
        self._strict = strict

    def generate(self,
                 sample: Union[np.ndarray, torch.Tensor, tf.Tensor],
                 bounds: List[List[int]] = None
                 ) -> tf.Tensor:
        noise = super(BoundedNoiseGeneratorTF, self).generate(sample)

        shape = sample.shape
        if shape[0] is None or shape[0] == 1:
            shape = shape[1:]

        if self._mask is None and bounds is None:
            bounds = [0] * len(shape) + list(shape)
            bounds = [bounds]
            self._set_mask(bounds)

        noise[0].assign(noise[0] * self._mask)
        self.apply_constraints(noise)
        return noise

    def apply_noise(self,
                    sample: Union[tf.Tensor, tf.Variable],
                    noise: List[tf.Variable]) -> tf.Tensor:
        return sample + (self._mask * noise[0])

    def set_bounds(self,
                   sample: Union[np.ndarray, tf.Tensor],
                   bounds: List[List[int]]
                   ) -> None:
        if self._mask is None:
            self._set_mask(sample, bounds)
        elif not self._strict and self._mask is not None:
            warnings.warn(
                "Bounds are already set. Overwriting bounds. Not running in strict mode may lead to unexpected behavior.")
            self._set_mask(sample, bounds)
        elif self._strict and self._mask is not None:
            raise AttributeError(
                "Bounds are already set. Create a new instance of BoundedNoiseGenerator to set new bounds.")

    def get_mask(self) -> Union[np.ndarray, tf.Tensor]:
        return self._mask

    def set_custom_mask(self,
                        sample: Union[np.ndarray, tf.Tensor],
                        mask: Union[np.ndarray, tf.Tensor],
                        threshold: float = 0.5
                        ) -> None:
        sample_shape = sample.shape
        mask_shape = mask.shape

        if len(sample_shape) != len(mask_shape):
            raise ValueError(f"Sample and mask must have the same number of dimensions. "
                             f"Got sample dimensions: {len(sample_shape)} and mask dimensions: {len(mask_shape)}")

        mask_np = mask.numpy() if isinstance(mask, tf.Tensor) else mask

        mask_resized = tf.image.resize(
            mask_np, sample_shape[:2], method="nearest").numpy()
        mask_normalized = (mask_resized - mask_resized.min()) / \
            (mask_resized.max() - mask_resized.min())
        mask_binary = (mask_normalized >= threshold).astype(np.uint8)
        self._mask = self.tensor_ops.to_tensor(mask_binary)

    def _set_mask(self,
                  sample: Union[np.ndarray, tf.Tensor],
                  bounds: List[List[int]]
                  ) -> None:
        shape = sample.shape
        if shape[0] is None or shape[0] == 1:
            shape = shape[1:]

        mask = np.zeros(shape, dtype=np.uint8)
        n_dims = len(shape)

        for bound in bounds:
            slices = tuple(
                slice(bound[i], bound[i] + bound[n_dims + i]) for i in range(n_dims))
            mask[slices] = 1

        self._mask = self.tensor_ops.to_tensor(mask)

    def _validate_bounds(self,
                         sample: Union[np.ndarray, tf.Tensor],
                         noise: List[tf.Tensor]
                         ) -> None:
        pass


class BoundedNoiseGeneratorTorch(AdditiveNoiseGeneratorTorch):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 use_constraints: bool,
                 epsilon: float,
                 scale: List[int] = None,
                 dist: Literal["normal", "uniform"] = "normal",
                 strict: bool = True
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
