from abc import ABC, abstractmethod
from typing import Literal, Union, List
import torch
import tensorflow as tf

class Optimizer(ABC):
    def __init__(self, framework: Literal["torch", "tf"], learning_rate: float = 0.001) -> None:
        self.framework = framework
        self.learning_rate = learning_rate
        
        if self.framework == "torch" and self.torch_apply is Optimizer.torch_apply:
            raise NotImplementedError("You must implement `torch_apply` for PyTorch models.")
        elif self.framework == "tf" and self.tensorflow_apply is Optimizer.tensorflow_apply:
            raise NotImplementedError("You must implement `tensorflow_apply` for TensorFlow models.")

    def apply(self, 
              model_weights: Union[List[torch.Tensor], List[tf.Variable]], 
              gradients: Union[List[torch.Tensor], List[tf.Tensor]]) -> None:
        if self.framework == "torch":
            if self.optimizer is not None and len(self.optimizer.param_groups[0]['params']) == 0:
                self.optimizer.add_param_group({'params': model_weights})
            if all(isinstance(w, torch.Tensor) for w in model_weights) and all(isinstance(g, torch.Tensor) for g in gradients):
                self.torch_apply(model_weights, gradients)
            else:
                raise ValueError("All model weights and gradients must be valid PyTorch tensors.")
        
        elif self.framework == "tf":
            if all(isinstance(w, tf.Variable) for w in model_weights) and all(isinstance(g, tf.Tensor) for g in gradients):
                self.tensorflow_apply(model_weights, gradients)
            else:
                raise ValueError("All model weights must be TensorFlow variables, and gradients must be TensorFlow tensors.")

    @abstractmethod
    def torch_apply(self, 
                    model_weights: torch.Tensor, 
                    gradients: torch.Tensor) -> None:
        raise NotImplementedError("You need to implement torch_apply for PyTorch models.")
    
    @abstractmethod
    def tensorflow_apply(self, 
                         model_weights: tf.Variable, gradients: tf.Tensor) -> None:
        raise NotImplementedError("You need to implement tensorflow_apply for TensorFlow models.")