from typing import Dict, Union, Literal

import torch
import numpy as np
import tensorflow as tf
from torch.nn import Module as TorchModel
from tensorflow.keras.models import Model as TFModel

from . import WhiteBoxAttack
from adversary.core.losses.loss_base import Loss
from adversary.core.noise_generators import NoiseGenerator
from adversary.core.optimizers.optimizer_base import Optimizer


class NonTargetedWhiteBoxAttack(WhiteBoxAttack):
    def __init__(self,
                 model: Union[TorchModel, TFModel],
                 loss: str | Loss,
                 optimizer: str | Optimizer,
                 optimizer_params: Dict | None = None,
                 noise_generator: NoiseGenerator = None,
                 preprocessing=None,
                 *args,
                 **kwargs) -> None:
        super().__init__(model, loss, optimizer, optimizer_params,
                         noise_generator, preprocessing, *args, **kwargs)

    def attack(self,
               sample: Union[np.ndarray, torch.Tensor, tf.Tensor],
               strategy: Literal['minimize', 'spread', 'uniform'] = 'minimize',
               epochs=10,
               *args,
               **kwargs):
        if strategy not in ['minimize', 'spread', 'uniform']:
            raise ValueError(
                "Invalid value for strategy. It must be 'ignore', 'distribute', or 'uniform'.")

        preprocessed_sample = self.preprocessing.preprocess(sample)
        noise = self.noise_generator.generate(preprocessed_sample)
        true_class = np.argmax(self.model.predict(preprocessed_sample), axis=1)

        num_classes = self.model_info["output_shape"][1]
        target_vector = np.zeros(shape=(num_classes, ))

        if strategy == 'minimize':
            target_vector = np.zeros(num_classes)
            target_vector[self.true_class] = 1e-6

        elif strategy == 'spread':
            target_vector = np.ones(num_classes) / (num_classes - 1)
            target_vector[self.true_class] = 1e-6
            target_vector /= target_vector.sum()

        elif strategy == 'uniform':
            target_vector = np.ones(num_classes) / num_classes

        target_vector = self.tensor_ops.to_tensor(target_vector)

        if len(target_vector) != num_classes:
            raise ValueError(
                "target_vector must be the same size as the outputs.")

        for _ in range(epochs):
            gradients = self.get_grads.calculate(
                self.model, preprocessed_sample, noise, self.noise_generator, target_vector)
            self.optimizer.apply([noise], [gradients])
            if self.noise_generator.use_constraints:
                noise = self.noise_generator.apply_constraints(noise)

            # Dev Only. Remove in Release
            print(self.model.predict(self.noise_generator.apply_noise(
                preprocessed_sample, noise), verbose=0)[0][true_class])

        return noise.numpy()
