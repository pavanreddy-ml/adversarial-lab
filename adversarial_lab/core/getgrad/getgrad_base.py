from abc import ABC, abstractmethod, ABCMeta
from typing import Literal, Union, List
import torch
import tensorflow as tf

class GetGradsMeta(ABCMeta):
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


class GetGradsBase(metaclass=GetGradsMeta):
    def __init__(self, 
                 model, 
                 loss):
        pass
    
    @abstractmethod
    def calculate(self, 
                  model: Union[torch.nn.Module, tf.keras.Model], 
                  inputs: Union[torch.Tensor, tf.Tensor], 
                  targets: Union[torch.Tensor, tf.Tensor]):
        pass

    def torch_op(self, 
                 model: torch.nn.Module, 
                 inputs: torch.Tensor, 
                 targets: torch.Tensor
                 ) -> List[torch.Tensor]:
        raise NotImplementedError()
    

    def tf_op(self, 
              model: tf.keras.Model, 
              inputs: tf.Tensor, 
              targets: tf.Tensor
              ) -> List[tf.Tensor]:
        raise NotImplementedError()
    

