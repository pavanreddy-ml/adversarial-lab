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


class PixelNoiseGeneratorTF(NoiseGenerator):
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 scale: Tuple[int] = (-1, 1),
                 n_pixels: int = 1,
                 pixel_selector: Literal["positive", "negative", "absolute"] = "absolute",
                 direction: Literal["positive", "negative", "bi"] = "positive",
                 strategy: Literal["single", "iterative", "incremental"] = "single",
                 reduce_dims: List[int] = None,
                 epsilon: float = np.inf,
                 bounds: List[List[int]] = None,
                 bounds_type: Literal["relative", "absolute"] = "relative",
                 custom_mask: Union[np.ndarray, tf.Tensor] = None,
                 custom_pixels: List[Tuple[int]] = None,
                 constraints: List[PostOptimizationConstraint] = None,
                 *args, **kwargs
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
        self.strategy = strategy
        self.custom_pixels = custom_pixels
        self.reduce_dims = reduce_dims
        self.epsilon = [-epsilon, epsilon]

        self.increment_per_iteration = kwargs.get("increment_per_iteration", 1)

        if self.custom_pixels is not None and self.n_pixels > len(self.custom_pixels): 
            raise ValueError(
                "Number of pixels should be less than or equal to the number of custom pixels")

        self._reset_optimization_meta()

    def generate_noise_meta(self,
                            sample: Union[np.ndarray, torch.Tensor, tf.Tensor],
                            bounds: List[List[int]] = None
                            ) -> tf.Variable:
        super().generate_noise_meta(sample, bounds)

        shape = sample.shape
        if shape[0] is None or shape[0] == 1:
            shape = shape[1:]

        noise = tf.zeros_like(sample)
        noise = tf.Variable(noise)

        self._reset_optimization_meta()

        noise.assign(noise * self._mask)
        self.apply_constraints(noise)

        return [noise]

    def get_noise(self,
                  noise_meta: List[tf.Tensor | tf.Variable]
                  ) -> np.ndarray:
        return tf.clip_by_value(noise_meta[0], self.scale[0], self.scale[1]).numpy()

    def apply_noise(self,
                    sample: tf.Tensor,
                    noise: tf.Variable
                    ) -> tf.Tensor:
        return tf.clip_by_value(sample + (noise[0] * self._mask), self.scale[0], self.scale[1])
    
    def apply_gradients(self,
                        tensor: tf.Variable,
                        gradients: tf.Tensor,
                        optimizer: Optimizer = None) -> None:
        if self.strategy == "single":
            self._optimize_single(tensor[0], gradients)
        elif self.strategy == "iterative":
            self._optimize_iterative(tensor, gradients, optimizer)
        elif self.strategy == "incremental":
            self._optimize_incremental(tensor[0], gradients)
        else:
            raise ValueError(f"Invalid Strategy: {self.strategy}")

    def _optimize_single(self,
                         tensor: List[tf.Variable],
                         gradients: tf.Tensor,
                         ) -> None:
        masked_gradients = self._get_masked_gradients(gradients, tensor[0])
        attacked_pixels = self._get_attacked_pixels(masked_gradients, tensor[0])
        tensor[0].assign(attacked_pixels)
        self.end_optimization = True

    def _optimize_iterative(self,
                            tensor: tf.Variable,
                            gradients: tf.Tensor,
                            optimizer: Optimizer,
                            ) -> None:

        if self.optimization_meta["iterative_mask"] is None:
            masked_gradients = self._get_masked_gradients(gradients, tensor[0])
            self.optimization_meta["top_k_mask"] = tf.reshape(masked_gradients, tf.shape(tensor[0]))
        masked_gradients = gradients * self.optimization_meta["top_k_mask"]
        optimizer.apply(tensor, masked_gradients)

    def _optimize_incremental(self,
                              tensor: tf.Variable,
                              gradients: tf.Tensor,
                              ) -> None:
        masked_gradients = self._get_masked_gradients(gradients, tensor[0])
        attacked_pixels = self._get_attacked_pixels(masked_gradients, tensor[0])
        tensor.assign(tf.where(attacked_pixels != 0, attacked_pixels, tensor))

    def _get_masked_gradients(self, 
                        gradients: tf.Tensor,
                        tensor: tf.Variable) -> tf.Tensor:
        flat_gradients = tf.reshape(gradients, [-1])

        if self.pixel_selector == "positive":
            flat_scores = flat_gradients
        elif self.pixel_selector == "negative":
            flat_scores = -flat_gradients
        elif self.pixel_selector == "absolute":
            flat_scores = tf.abs(flat_gradients)
        else:
            raise ValueError(f"Invalid direction: {self.pixel_selector}")

        _, top_k_indices = tf.math.top_k(flat_scores, k=self.increment_per_iteration)

        top_k_mask = tf.scatter_nd(
            indices=tf.expand_dims(top_k_indices, axis=1),
            updates=tf.ones_like(top_k_indices, dtype=tf.float32),
            shape=tf.shape(flat_gradients),
        )

        self.optimization_meta["top_k_mask"] = tf.reshape(top_k_mask, tf.shape(tensor))

        masked_gradients = flat_gradients * top_k_mask

        return masked_gradients
    
    def _get_attacked_pixels(self, masked_gradients: tf.Tensor, tensor: tf.Tensor) -> tf.Tensor:
        if self.direction == "positive":
            updated_gradients = tf.where(
                tf.not_equal(masked_gradients, 0),
                tf.fill(tf.shape(masked_gradients), tf.cast(self.epsilon[1], dtype=masked_gradients.dtype)),
                masked_gradients
            )
        elif self.direction == "negative":
            updated_gradients = tf.where(
                tf.not_equal(masked_gradients, 0),
                tf.fill(tf.shape(masked_gradients), tf.cast(self.epsilon[0], dtype=masked_gradients.dtype)),
                masked_gradients
            )
        elif self.direction == "bi":
            updated_gradients = tf.where(
                masked_gradients > 0,
                tf.fill(tf.shape(masked_gradients), tf.cast(self.epsilon[0], dtype=masked_gradients.dtype)),
                tf.where(
                    masked_gradients < 0,
                    tf.fill(tf.shape(masked_gradients), tf.cast(self.epsilon[1], dtype=masked_gradients.dtype)),
                    masked_gradients
                )
            )
        else:
            raise ValueError(f"Invalid direction: {self.direction}")
        
        return tf.reshape(updated_gradients, tf.shape(tensor))
    
    def _reset_optimization_meta(self):
        self.optimization_meta = {
            "iterative_mask": None,
        }


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
