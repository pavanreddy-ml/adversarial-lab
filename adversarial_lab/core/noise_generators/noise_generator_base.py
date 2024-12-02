from abc import ABC, abstractmethod, ABCMeta
from typing import Literal, Union, List, Tuple
import numpy as np
import torch
import warnings
import tensorflow as tf
import importlib
from adversarial_lab.core.optimizers import Optimizer
from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.constraints import PostOptimizationConstraint


class NoiseGeneratorMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        framework = kwargs.get('framework', None)
        if framework is None and len(args) > 0:
            framework = args[0]

        base_class_name = cls.__name__.replace("NoiseGenerator", "")
        module_name = f".{base_class_name.lower()}_noise_generator"

        if framework == "torch":
            specific_class_name = f"{base_class_name}NoiseGeneratorTorch"
        elif framework == "tf":
            specific_class_name = f"{base_class_name}NoiseGeneratorTF"
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        try:
            module = importlib.import_module(module_name, package=__package__)
            specific_class = getattr(module, specific_class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Class {specific_class_name} not found in module {module_name}. Ensure it is defined.") from e

        instance = specific_class(*args, **kwargs)
        return instance
    
class NoiseGenerator(ABC):
    def __init__(self, 
                 framework: Literal["torch", "tf"],
                 bounds: List[List[int]] = None,
                 bounds_type: Literal["relative", "absolute"] = "relative",
                 custom_mask: Union[np.ndarray, tf.Tensor] = None,
                 constraints: List[PostOptimizationConstraint] = None
                 ) -> None:
        self.framework = framework

        self.tensor_ops = TensorOps(framework)

        self._mask = None
        self._bounds = bounds
        self._bounds_type = bounds_type
        self.custom_mask = custom_mask

        self._constraints = constraints if constraints is not None else []
        if not all(isinstance(constraint, PostOptimizationConstraint) for constraint in self._constraints):
            raise ValueError("All constraints must be of type POConstraint")

    def generate_noise_meta(self, 
                            sample,
                            *args, 
                            **kwargs
                            ) -> List[tf.Variable | torch.Tensor]:
        if not isinstance(sample, (np.ndarray, torch.Tensor, tf.Tensor)):
            raise TypeError("Input must be of type np.ndarray, tf.Tensor")
        
        self._apply_bounds(sample)

    @abstractmethod
    def get_noise(self,
                  noise_meta: Union[tf.Variable | torch.Tensor]
                  ) -> np.ndarray:
        pass

    @abstractmethod
    def apply_noise(self, 
                    noise_meta: Union[tf.Variable | torch.Tensor],
                    *args, 
                    **kwargs) -> Union[tf.Tensor | tf.Variable | torch.Tensor]:
        pass

    def apply_gradients(self, 
                        tensor: tf.Variable, 
                        gradients: tf.Tensor,
                        optimizer: Optimizer
                        ) -> None:
        optimizer.apply(tensor, gradients)

    def set_bounds(self, 
                   bounds: List[List[int]]
                   ) -> None:
        self._bounds = bounds
    
    def _apply_bounds(self,
                   sample: Union[np.ndarray, tf.Tensor],
                   ) -> None:
        if self.custom_mask is not None:
            self._set_custom_mask(sample, self.custom_mask)
            return
        
        shape = sample.shape
        if shape[0] is None or shape[0] == 1:
            shape = shape[1:]

        if self._bounds is None:
            default_bounds = [0] * len(shape) + list(shape)
            self._bounds = [default_bounds]

        self._validate_bounds(shape, self._bounds)
        self._set_mask(sample, self._bounds)

    def _set_mask(self,
                  sample: Union[np.ndarray, tf.Tensor],
                  bounds: List[List[int]] = None
                  ) -> None:
        if bounds is None:
            mask = np.ones_like(sample.shape)
        else:
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

    def _set_custom_mask(self,
                        sample: Union[np.ndarray, tf.Tensor],
                        mask: np.ndarray,
                        threshold: float = 0.5
                        ) -> None:
        sample_shape = sample.shape
        mask_shape = mask.shape

        if len(sample_shape) != len(mask_shape):
            raise ValueError(f"Sample and mask must have the same number of dimensions. "
                             f"Got sample dimensions: {len(sample_shape)} and mask dimensions: {len(mask_shape)}")
        
        mask_normalized = (mask - mask.min()) / (mask.max() - mask.min())
        mask_binary = (mask_normalized >= threshold).astype(np.uint8)
        self._mask = self.tensor_ops.to_tensor(mask_binary)

    def _validate_bounds(self,
                         sample_shape: List[int],
                         bounds: List[Tuple[int]]
                         ) -> None:
        if any(2*len(sample_shape) != len(bound) for bound in bounds):
            raise ValueError(
                "Number of dimensions in bounds should be equal to the number of dimensions in the sample.")
        
        for bound in bounds:
            for i in range(len(sample_shape)):
                if bound[i] > sample_shape[i]:
                    raise ValueError(f"Bound should be less than the corresponding dimension in the sample: Sample shape -> {sample_shape}, Bound -> {bound}")

    def get_mask(self) -> np.ndarray:
        return self._mask
    
    def apply_constraints(self, noise, *args, **kwargs):
        for constraint in self._constraints:
            constraint.apply(noise)
