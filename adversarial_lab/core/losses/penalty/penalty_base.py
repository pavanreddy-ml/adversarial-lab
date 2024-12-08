from abc import ABC, abstractmethod, ABCMeta
from typing import Literal
import torch
import tensorflow as tf
import warnings
import traceback


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
    def __init__(self, 
                 framework: Literal["torch", "tf"],
                 *args,
                 **kwargs) -> None:
        self.framework = framework

        self.value = None
        self.warned = False

    @abstractmethod
    def calculate(self, *args, **kwargs):
        pass

    def torch_op(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("torch_op not implemented")

    def tf_op(self, *args, **kwargs) -> tf.Tensor:
        raise NotImplementedError("tf_op not implemented")
    
    def set_value(self, value):
        try:
            if self.framework == "torch":
                self.value = value.item()
            elif self.framework == "tf":
                self.value = float(value.numpy())
        except Exception as e:
            if not self.warned:
                self.warned = True
                warnings.warn(f"Error while setting value: {e}. Traceback: {traceback.format_exc()}")

            self.value = None

    def __repr__(self) -> str:
        return self.__class__.__name__
