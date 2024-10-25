from abc import ABC, abstractmethod, ABCMeta
from typing import Literal, Union, List
import numpy as np
import torch
import tensorflow as tf
import importlib
from adversarial_lab.core.optimizers import Optimizer


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

        instance = super(NoiseGeneratorMeta, specific_class).__call__(*args, **kwargs)

        if framework == "torch":
            instance.apply_gradients = instance.apply_gradients_torch
        elif framework == "tf":
            instance.apply_gradients = instance.apply_gradients_tf

        return instance
    
class NoiseGenerator(ABC):
    def __init__(self, 
                 framework: Literal["torch", "tf"],
                 use_constraints: bool):
        self.framework = framework
        self.use_constraints = use_constraints

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    @abstractmethod
    def apply_noise(self, *args, **kwargs):
        pass

    @abstractmethod
    def apply_constraints(self, *args, **kwargs):
        pass

    def apply_gradients(self, 
                        tensor: tf.Variable, 
                        gradients: tf.Tensor,
                        optimizer: Optimizer
                        ) -> None:
        pass
    
    def get_noise(self, 
                  noise_components: List[tf.Tensor | tf.Variable]
                  ) -> np.ndarray:
        noise = tf.add_n(noise_components)
        return noise.numpy()

    def apply_gradients_tf(self, 
                        tensor: List[tf.Variable], 
                        gradients: List[tf.Variable],
                        optimizer: Optimizer
                        ) -> None:
        optimizer.apply(tensor, gradients)

    def apply_gradients_torch(self, 
                        tensor: torch.Tensor, 
                        gradients: torch.Tensor,
                        optimizer: Optimizer
                        ) -> None:
        raise NotImplementedError("apply_gradients_torch is not implemented yet.")