from abc import ABC, abstractmethod, ABCMeta
from typing import Literal
import torch
import tensorflow as tf


class NoiseGeneratorMeta(ABCMeta):
    def __new__(cls, name, bases, dct):
        if 'generate' not in dct:
            raise TypeError(f"{name} class must implement a 'generate' method.")
        
        original_generate = dct['generate']

        def wrapped_run(self, *args, **kwargs):
            if not hasattr(self, 'framework'):
                raise AttributeError(f"Instance of {name} class must have a 'framework' attribute.")
            
            if self.framework == "torch":
                return self.torch_op(*args, **kwargs)
            elif self.framework == "tf":
                return self.tf_op(*args, **kwargs)
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")
        
        dct['generate'] = wrapped_run
        return super().__new__(cls, name, bases, dct)


class NoiseGenerator(metaclass=NoiseGeneratorMeta):
    def __init__(self, framework: Literal["torch", "tf"]) -> None:
        self.framework = framework

    @abstractmethod
    def generate(self):
        pass

    def torch_op(self) -> torch.Tensor:
        raise NotImplementedError("torch_op not implemented")

    def tf_op(self) -> tf.Tensor:
        raise NotImplementedError("tf_op not implemented")
