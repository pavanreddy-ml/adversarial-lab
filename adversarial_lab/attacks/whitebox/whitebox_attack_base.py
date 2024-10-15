from abc import ABC, abstractmethod
from typing import Union, Any, Dict

import numpy as np
from copy import deepcopy

import torch
import tensorflow as tf
from torch.nn import Module as TorchModel
from tensorflow.keras.models import Model as TFModel

from adversarial_lab.core.getgrad import GetGrads
from adversarial_lab.core.modelinfo import ModelInfo
from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.losses import Loss, LossRegistry
from adversarial_lab.core.optimizers import Optimizer, OptimizerRegistry
from adversarial_lab.core.preprocessing import NoPreprocessing, Preprocessing
from adversarial_lab.core.noise_generators import AdditiveNoiseGenerator, NoiseGenerator

class WhiteBoxAttack(ABC):
    def __init__(self, 
                 model: Union[TorchModel, TFModel],
                 loss: Union[str, Loss],
                 optimizer: Union[str, Optimizer],
                 noise_generator: NoiseGenerator = None,
                 preprocessing: Preprocessing = None,
                 *args,
                 **kwargs
                 ) -> None:
        
        if isinstance(model, torch.nn.Module):
            self.framework = "torch"
        elif isinstance(model, tf.keras.Model):
            self.framework = "tf"
        else:
            raise ValueError("Unsupported model type. Must be a PyTorch or TensorFlow model.")
        
        get_model_info = ModelInfo(framework=self.framework)
        self.model_info = get_model_info.get_info(model)
        self.model = model

        self._optimizer_arg = optimizer
        self._initialize_optimizer(self._optimizer_arg)
        self._initialize_loss(loss)
        self._initialize_noise_generator(noise_generator)
        self._initialize_preprocessing(preprocessing)

        self.tensor_ops = TensorOps(framework=self.framework)
        self.get_grads = GetGrads(framework=self.framework, loss=self.loss)
        
    def attack(self,
               *args,
               **kwargs
               ) -> np.ndarray:
        self._initialize_optimizer(self._optimizer_arg)

    def _initialize_optimizer(self, optimizer):
        optimizer_copy = deepcopy(optimizer)

        if isinstance(optimizer_copy, str):
            optimizer_class = OptimizerRegistry.get(optimizer_copy)
            self.optimizer = optimizer_class(framework=self.framework)
        elif isinstance(optimizer_copy, Optimizer):
            self.optimizer = optimizer_copy
        else:
            raise TypeError(f"Invalid type for optimizer: '{type(optimizer_copy)}'")

    def _initialize_loss(self, loss):
        if isinstance(loss, str):
            loss_class = LossRegistry.get(loss)
            self.loss = loss_class(framework=self.framework)
        elif isinstance(loss, Loss):
            self.loss = loss
        else:
            raise TypeError(f"Invalid type for loss: '{type(loss)}'")
        
    def _initialize_noise_generator(self, noise_generator):
        if noise_generator is None:
            self.noise_generator = AdditiveNoiseGenerator(framework=self.framework, use_constraints=True, epsilon=0.005, dist='uniform')
        elif isinstance(noise_generator, NoiseGenerator):
            self.noise_generator = noise_generator
        else:
            raise TypeError(f"Invalid type for noise_generator: '{type(noise_generator)}'")
        
    def _initialize_preprocessing(self, preprocessing):
        if preprocessing is None:
            self.preprocessing = NoPreprocessing()
        elif isinstance(preprocessing, Preprocessing):
            self.preprocessing = preprocessing
        else:
            raise TypeError(f"Invalid type for preprocessing: '{type(preprocessing)}'")