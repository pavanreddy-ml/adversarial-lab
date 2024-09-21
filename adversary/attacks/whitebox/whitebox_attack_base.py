from abc import ABC, abstractmethod

from typing import Union, Any, Dict

import numpy as np
import torch
from torch.nn import Module as TorchModel
from tensorflow.keras.models import Model as TFModel
import tensorflow as tf

from adversary.models.losses import Loss, LossRegistry
from adversary.models.optimizers import Optimizer, OptimizerRegistry

class WhiteBoxAttack(ABC):
    def __init__(self, 
                 model: Union[TorchModel, TFModel],
                 loss: Union[str, Loss],
                 optimizer: Union[str, Optimizer],
                 optimizer_params: Union[Dict, None] = None,
                 *args,
                 **kwargs
                 ) -> None:
        
        if optimizer_params is None: self.optimizer_params = {}

        if isinstance(model, torch.nn.Module):
            self.framework = "torch"
        elif isinstance(model, tf.keras.Model):
            self.framework = "tensorflow"
        else:
            raise ValueError("Unsupported model type. Must be a PyTorch or TensorFlow model.")
        
        if isinstance(loss, str):
            self.loss = LossRegistry.get(loss)
        elif isinstance(loss, Loss):
            self.loss = loss
        else:
            raise TypeError(f"Invalid type for loss: '{type(loss)}'")
        
        if isinstance(optimizer, str):
            optimizer_class = OptimizerRegistry.get(optimizer)
            self.optimizer = optimizer_class(**optimizer_params)
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise TypeError(f"Invalid type for optimizer: '{type(optimizer)}'")

    @abstractmethod
    def attack(self, 
               sample: Union[np.ndarray, torch.Tensor, tf.Tensor],
               confidence_threshold: float,
               *args,
               **kwargs
               ) -> np.ndarray:
        pass