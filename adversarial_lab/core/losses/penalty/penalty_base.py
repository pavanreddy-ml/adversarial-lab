from abc import ABC, abstractmethod, ABCMeta
from typing import Literal
import torch
import tensorflow as tf


class PenaltyMeta(ABCMeta):
    def __new__(cls, name, bases, dct):
        if 'calculate' not in dct:
            raise TypeError(f"{name} class must implement a 'calculate' method.")
        
        original_calculate = dct['calculate']

        def wrapped_run(self, *args, **kwargs):
            if not hasattr(self, 'framework'):
                raise AttributeError(f"Instance of {name} class must have a 'framework' attribute.")
            
            if self.framework == "torch":
                return self.torch_op(*args, **kwargs)
            elif self.framework == "tf":
                return self.tf_op(*args, **kwargs)
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")
        
        dct['calculate'] = wrapped_run
        return super().__new__(cls, name, bases, dct)


class Penalty(metaclass=PenaltyMeta):
    def __init__(self, framework: Literal["torch", "tf"]) -> None:
        self.framework = framework
        self.value = None

    @abstractmethod
    def calculate(self, loss):
        pass

    def torch_op(self, loss) -> torch.Tensor:
        raise NotImplementedError("torch_op not implemented")

    def tf_op(self, loss) -> tf.Tensor:
        raise NotImplementedError("tf_op not implemented")
    
    def set_value(self, value):
        if self.framework == "torch":
            self.value = value.item()
        elif self.framework == "tf":
            self.value = float(value.numpy())
