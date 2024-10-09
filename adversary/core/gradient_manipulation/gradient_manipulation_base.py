from abc import ABC, abstractmethod, ABCMeta
from typing import Literal, Union, List
import torch
import tensorflow as tf


class GradientManipulationMeta(ABCMeta):
    def __new__(cls, name, bases, dct):
        if 'apply' not in dct:
            raise TypeError(f"{name} class must implement a 'apply' method.")

        original_apply = dct['apply']

        def wrapped_run(self, *args, **kwargs):
            if not hasattr(self, 'framework'):
                raise AttributeError(
                    f"Instance of {name} class must have a 'framework' attribute.")

            if self.framework == "torch":
                return self.torch_op(*args, **kwargs)
            elif self.framework == "tf":
                return self.tf_op(*args, **kwargs)
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")

        dct['apply'] = wrapped_run
        return super().__new__(cls, name, bases, dct)


class GradientManipulation(metaclass=GradientManipulationMeta):
    def __init__(self, framework: Literal["torch", "tf"]) -> None:
        self.framework = framework

    @abstractmethod
    def apply(self,
              gradients: torch.Tensor):
        pass

    def torch_op(self,
                 gradients: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("torch_op not implemented")

    def tf_op(self,
              gradients: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError("tf_op not implemented")
