from abc import ABC, abstractmethod
from typing import Union, Any, Dict

import torch
import numpy as np
import tensorflow as tf
from torch.nn import Module as TorchModel
from tensorflow.keras.models import Model as TFModel

from adversary.core.losses import Loss, LossRegistry
from adversary.core.get_grad import GetGrads
from adversary.core.modelinfo import ModelInfo
from adversary.core.tensor_ops import TensorOps
from adversary.core.optimizers import Optimizer, OptimizerRegistry
from adversary.core.preprocessing import NoPreprocessing, Preprocessing
from adversary.core.noise_generators import AdditiveNoiseGenerator, NoiseGenerator

class WhiteBoxAttack(ABC):
    def __init__(self, 
                 model: Union[TorchModel, TFModel],
                 loss: Union[str, Loss],
                 optimizer: Union[str, Optimizer],
                 optimizer_params: Union[Dict, None] = None,
                 noise_generator: NoiseGenerator = None,
                 preprocessing: Preprocessing = None,
                 *args,
                 **kwargs
                 ) -> None:
        
        self.model = model
        
        if optimizer_params is None: 
            self.optimizer_params = {}

        if isinstance(model, torch.nn.Module):
            self.framework = "torch"
        elif isinstance(model, tf.keras.Model):
            self.framework = "tf"
        else:
            raise ValueError("Unsupported model type. Must be a PyTorch or TensorFlow model.")
        
        if isinstance(loss, str):
            loss_class = LossRegistry.get(loss)
            self.loss = loss_class(framework=self.framework)
        elif isinstance(loss, Loss):
            self.loss = loss
        else:
            raise TypeError(f"Invalid type for loss: '{type(loss)}'")
        
        if isinstance(optimizer, str):
            optimizer_class = OptimizerRegistry.get(optimizer)
            self.optimizer = optimizer_class(framework=self.framework, **self.optimizer_params)
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise TypeError(f"Invalid type for optimizer: '{type(optimizer)}'")
        
        if noise_generator is None:
            self.noise_generator = AdditiveNoiseGenerator(framework="tf", use_constraints=True, epsilon=0.005, dist='uniform')
        elif isinstance(noise_generator, NoiseGenerator):
            self.noise_generator = noise_generator
        else:
            raise TypeError(f"Invalid type for optimizer: '{type(optimizer)}'")
        
        if preprocessing is None:
            self.preprocessing = NoPreprocessing()
        elif isinstance(preprocessing, Preprocessing):
            self.preprocessing = preprocessing
        else:
            raise TypeError(f"Invalid type for preprocessing: '{type(preprocessing)}'")

        self.tensor_ops = TensorOps(framework=self.framework)

        self.get_grads = GetGrads(framework=self.framework, loss=self.loss)
        
        get_model_info = ModelInfo(framework=self.framework)
        self.model_info = get_model_info.get_info(model)

    @abstractmethod
    def attack(self,
               sample,
               epochs=10,
               *args,
               **kwargs):
        pass