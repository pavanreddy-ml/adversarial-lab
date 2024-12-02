from typing import Literal, List, Union, TypeVar, Generic, Dict, Any, Tuple, Union
import warnings

import torch
import numpy as np
import tensorflow as tf
from torch.nn import functional as F

from . import NoiseGenerator, NoiseGeneratorMeta
from adversarial_lab.core.optimizers import Optimizer
from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.constraints import PostOptimizationConstraint


class PixelNoiseGeneratorTF(NoiseGenerator, metaclass=NoiseGeneratorMeta):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 scale: Tuple[int] = (-1, 1),
                 n_pixels: int = 1,
                 pixel_selector: Literal["positive",
                                         "negative", "absolute"] = "absolute",
                 direction: Literal["positive", "negative", "bi"] = "positive",
                 gradient_application: Literal["individual",
                                               "mean"] = "individual",
                 strategy: Literal["single",
                                   "iterative", "adaptive"] = "single",
                 along_dims: List[Tuple[Tuple[bool], int, str]] = None,
                 bounds: List[List[int]] = None,
                 bounds_type: Literal["relative", "absolute"] = "relative",
                 custom_mask: Union[np.ndarray, tf.Tensor] = None,
                 custom_pixels: List[Tuple[int]] = None,
                 constraints: List[PostOptimizationConstraint] = None
                 ) -> None:
        super().__init__(framework=framework,
                         bounds=bounds,
                         bounds_type=bounds_type,
                         custom_mask=custom_mask,
                         constraints=constraints)

        self.n_pixels = n_pixels
        self.scale = scale
        self.pixel_selector = pixel_selector
        self.direction = direction
        self.gradient_application = gradient_application
        self.strategy = strategy
        self.along_dims = along_dims
        self.custom_pixels = custom_pixels

        if self.n_pixels > len(self.custom_pixels):
            raise ValueError(
                "Number of pixels should be less than or equal to the number of custom pixels")

        self.optimization_meta = {}

    def generate_noise_meta(self,
                            sample: Union[np.ndarray, torch.Tensor, tf.Tensor],
                            bounds: List[List[int]] = None
                            ) -> tf.Variable:
        if not isinstance(sample, (np.ndarray, torch.Tensor, tf.Tensor)):
            raise TypeError(
                "Input must be of type np.ndarray, tf.Tensor")

        shape = sample.shape
        if shape[0] is None or shape[0] == 1:
            shape = shape[1:]

        noise = tf.zeros_like(sample)
        noise = tf.Variable(noise)

        self._validate_along_dims(shape)

        self._set_bounds(sample, bounds)
        self._set_mask(bounds)

        noise.assign(noise * self._mask)
        self.apply_constraints(noise)

        return noise

    def get_noise(self,
                  noise_meta: Union[tf.Tensor | tf.Variable]
                  ) -> np.ndarray:
        return noise_meta.numpy()

    def apply_noise(self,
                    sample: tf.Tensor,
                    noise: tf.Variable
                    ) -> tf.Tensor:
        # return sample + (noise * self._mask * (self.optimization_meta["pixel_mask"] if self.optimization_meta.get("pixel_mask") is not None else 1))
        return sample + (noise * self._mask)
    
    def apply_gradients(self,
                        tensor: tf.Variable,
                        gradients: tf.Tensor,
                        optimizer: Optimizer = None) -> None:
        for i, ad in enumerate(self.along_dims):
            # Get Gradient Mask
            # Configure New Gradients.
            self._optimize_noise(tensor, gradients, optimizer, ad)

    def _optimize_noise(self,
                        tensor: tf.Variable,
                        gradients: tf.Tensor,
                        optimizer: Optimizer,
                        dim_info: Tuple[Tuple[bool], int, str]
                        ) -> None:
        if self.strategy == "single":
            self._optimize_single(tensor, gradients, optimizer, dim_info)
        elif self.strategy == "iterative":
            self._optimize_iterative(tensor, gradients, optimizer, dim_info)
        elif self.strategy == "adaptive":
            self._optimize_adaptive(tensor, gradients, optimizer, dim_info)
        elif self.strategy == "incremental":
            self._optimize_incremental(tensor, gradients, optimizer, dim_info)
        else:
            raise ValueError(f"Invalid Strategy: {self.strategy}")

    def _optimize_single(self,
                         tensor: tf.Variable,
                         gradients: tf.Tensor,
                         optimizer: Optimizer,
                         dim_info: Tuple[Tuple[bool], int, str]
                         ) -> None:
        raise NotImplementedError("Single Optimization Not Implemented")

    def _optimize_iterative(self,
                            tensor: tf.Variable,
                            gradients: tf.Tensor,
                            optimizer: Optimizer,
                            dim_info: Tuple[Tuple[bool], int, str]
                            ) -> None:
        raise NotImplementedError("Iterative Optimization Not Implemented")

    def _optimize_adaptive(self,
                           tensor: tf.Variable,
                           gradients: tf.Tensor,
                           optimizer: Optimizer,
                           dim_info: Tuple[Tuple[bool], int, str]
                           ) -> None:
        raise NotImplementedError("Adaptive Optimization Not Implemented")

    def _optimize_incremental(self,
                              tensor: tf.Variable,
                              gradients: tf.Tensor,
                              optimizer: Optimizer,
                              dim_info: Tuple[Tuple[bool], int, str]
                              ) -> None:
        raise NotImplementedError("Incremental Optimization Not Implemented")

    # def _optimize_noise(self,
    #                     tensor: tf.Variable,
    #                     gradients: tf.Tensor,
    #                     optimizer: Optimizer,
    #                     dim_info: Tuple[Tuple[bool], int, str]
    #                     ) -> None:
    #     along_dim, n_pixels, direction = dim_info
    #     input_shape = tf.shape(gradients)

    #     reduction_axes = [i for i, flag in enumerate(along_dim) if not flag]
    #     reduced_gradients = tf.reduce_sum(
    #         gradients, axis=reduction_axes, keepdims=True)
    #     flat_gradients = tf.reshape(reduced_gradients, [-1])

    #     if direction == "positive":
    #         flat_scores = flat_gradients
    #     elif direction == "negative":
    #         flat_scores = -flat_gradients
    #     elif direction == "bi":
    #         flat_scores = tf.abs(flat_gradients)
    #     else:
    #         raise ValueError(f"Invalid direction: {direction}")

    #     top_k_values, top_k_indices = tf.math.top_k(flat_scores, k=n_pixels)

    #     flat_mask = tf.scatter_nd(
    #         tf.expand_dims(top_k_indices, axis=1),
    #         tf.ones_like(top_k_values, dtype=tf.float32),
    #         shape=tf.shape(flat_gradients)
    #     )

    #     mask_reduced = tf.reshape(flat_mask, tf.shape(reduced_gradients))
    #     broadcast_mask = tf.broadcast_to(mask_reduced, input_shape)
    #     masked_gradients = gradients * broadcast_mask
    #     if optimizer:
    #         optimizer.apply([(masked_gradients, tensor)])
    #     else:
    #         tensor.assign_add(masked_gradients)

    def _validate_along_dims(self, shape):
        if self.along_dims is None:
            self.along_dims = [
                ((True for i in range(len(shape))), self.n_pixels, self.direction)]
        else:
            for dim in self.along_dims:
                if dim[1] >= len(shape):
                    raise ValueError("Invalid Dimension")


class PixelNoiseGeneratorTorch(NoiseGenerator):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 epsilon: float,
                 scale: List[int] = None,
                 dist: Literal["normal", "uniform"] = "normal"
                 ) -> None:
        pass

    def generate(self,
                 shape: List[int]
                 ) -> List[torch.Tensor]:
        pass

    def apply_noise(self, *args, **kwargs):
        raise NotImplementedError("Apply Noise Not Implemented for Torch")

    def apply_constraints(self, *args, **kwargs):
        raise NotImplementedError("Apply Constrains Not Implemented for Torch")


class PixelNoiseGenerator(NoiseGenerator, metaclass=NoiseGeneratorMeta):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 use_constraints: bool,
                 epsilon: float,
                 scale: List[int] = None,
                 dist: Literal["normal", "uniform"] = "normal") -> None:
        self.framework: Literal["torch", "tf"] = framework
        self.use_constraints: bool = use_constraints
