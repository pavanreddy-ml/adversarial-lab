from abc import ABC, abstractmethod, ABCMeta
from typing import Literal, Union, List
import numpy as np
import torch
import tensorflow as tf
import importlib


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

        return super(NoiseGeneratorMeta, specific_class).__call__(*args, **kwargs)
    
class NoiseGenerator(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    @abstractmethod
    def apply_noise(self, *args, **kwargs):
        pass

    @abstractmethod
    def apply_constraints(self, *args, **kwargs):
        pass