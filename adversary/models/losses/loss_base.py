from abc import ABC, abstractmethod
from typing import Literal
import torch
import tensorflow as tf

class Loss(ABC):
    def __init__(self, framework: Literal["torch", "tf"]) -> None:
        self.framework = framework
        
        if self.framework == "torch" and self.torch_loss is Loss.torch_loss:
            raise NotImplementedError("You must implement `torch_loss` for PyTorch models.")
        elif self.framework == "tf" and self.tensorflow_loss is Loss.tensorflow_loss:
            raise NotImplementedError("You must implement `tensorflow_loss` for TensorFlow models.")

    def calculate(self, output: torch.Tensor | tf.Tensor, target: torch.Tensor | tf.Tensor) -> torch.Tensor | tf.Tensor:
        if self.framework == "torch":
            if isinstance(output, torch.Tensor):
                return self.torch_loss(output, target)
            else:
                raise ValueError("The provided output is not a valid PyTorch tensor.")
        
        elif self.framework == "tf":
            if isinstance(output, tf.Tensor):
                return self.tensorflow_loss(output, target)
            else:
                raise ValueError("The provided output is not a valid TensorFlow tensor.")
    
    @abstractmethod
    def torch_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("You need to implement torch_loss for PyTorch models.")
    
    @abstractmethod
    def tensorflow_loss(self, output: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError("You need to implement tensorflow_loss for TensorFlow models.")