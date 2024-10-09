from typing import Dict, Union
from torch.nn import Module as TorchModel
from adversary.core.losses.loss_base import Loss
from adversary.core.optimizers.optimizer_base import Optimizer
from adversary.core.noise_generators import NoiseGenerator
from tensorflow.keras.models import Model as TFModel
import tensorflow as tf
from . import WhiteBoxAttack
import torch

import numpy as np


class TargetedWhiteBoxAttack(WhiteBoxAttack):
    def __init__(self, 
                 model: Union[TorchModel, TFModel], 
                 loss: str | Loss, 
                 optimizer: str | Optimizer, 
                 optimizer_params: Dict | None = None, 
                 noise_generator: NoiseGenerator = None,
                 preprocessing = None,
                 *args, 
                 **kwargs) -> None:
        super().__init__(model, loss, optimizer, optimizer_params, noise_generator,preprocessing, *args, **kwargs)

    def attack(self,
               sample: Union[np.ndarray, torch.Tensor, tf.Tensor],
               target_class: int,
               target_vector=None,
               epochs=10,
               *args,
               **kwargs):
        if target_vector is None:
            if target_class >= self.model_info["output_shape"][1]:
                raise ValueError("target_class exceeds the dimension of the outputs.")
            target_vector = np.zeros(shape=(self.model_info["output_shape"][1], ))
            target_vector[target_class] = 1
            target_vector = self.tensor_ops.to_tensor(target_vector)
        else:
            target_vector = self.tensor_ops.to_tensor(target_vector)

        if len(target_vector) != self.model_info["output_shape"][1]:
            raise ValueError("target_vector must be the same size outputs.")
        
        preprocessed_sample = self.preprocessing.preprocess(sample)
        noise = self.noise_generator.generate(preprocessed_sample)

        for _ in range(epochs):
            gradients = self.get_grads.calculate(self.model, preprocessed_sample, noise, self.noise_generator, target_vector)
            self.optimizer.apply([noise], [gradients])
            if self.noise_generator.use_constraints:
                noise = self.noise_generator.apply_constraints(noise)

            # Dev Only. Remove in Release
            print(self.model.predict(noise, verbose=0)[0][target_class])

        return noise.numpy()
        


        

        