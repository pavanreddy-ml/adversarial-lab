from typing import Dict, Union

import random
import numpy as np
from tqdm import tqdm

import torch
import tensorflow as tf
from torch.nn import Module as TorchModel
from tensorflow.keras.models import Model as TFModel

from . import WhiteBoxAttack
from adversarylab.core.losses.loss_base import Loss
from adversarylab.core.noise_generators import NoiseGenerator
from adversarylab.core.optimizers.optimizer_base import Optimizer


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
        if target_class == -1:
            target_class = random.randint(0, self.model_info["output_shape"][1] - 1)

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
            print(self.model.predict(self.noise_generator.apply_noise(preprocessed_sample, noise), verbose=0)[0][target_class])

        return noise.numpy()
        


        

        