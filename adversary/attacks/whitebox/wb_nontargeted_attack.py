from typing import Dict, Union, Literal

import random
import numpy as np
from tqdm import tqdm

import torch
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
               strategy: Literal['minimize', 'spread', 'uniform', 'random'] = 'minimize',
               epochs=10,
               *args,
               **kwargs):
        if strategy not in ['minimize', 'spread', 'uniform', 'random']:
            raise ValueError(
                "Invalid value for strategy. It must be 'ignore', 'distribute', or 'uniform'.")

        preprocessed_sample = self.preprocessing.preprocess(sample)
        noise = self.noise_generator.generate(preprocessed_sample)
        true_class = np.argmax(self.model.predict(
            preprocessed_sample), axis=1)[0]
        
        predictions = self.model.predict(
                self.noise_generator.apply_noise(preprocessed_sample, noise), verbose=0)
        target_vector = self._get_target_vector(
            predictions, true_class, strategy)
        
        random_class = np.argmax(target_vector, axis=1)[0]

        for _ in range(epochs):
            if strategy == 'minimize':
                predictions = self.model.predict(
                    self.noise_generator.apply_noise(preprocessed_sample, noise), verbose=0)
                target_vector = self._get_target_vector(
                    predictions, true_class, strategy)

            gradients = self.get_grads.calculate(
                self.model, preprocessed_sample, noise, self.noise_generator, target_vector)
            self.optimizer.apply([noise], [gradients])
            if self.noise_generator.use_constraints:
                noise = self.noise_generator.apply_constraints(noise)

            # Dev Only. Remove in Release
            print(self.model.predict(self.noise_generator.apply_noise(
                preprocessed_sample, noise), verbose=0)[0][true_class])
            print(self.model.predict(self.noise_generator.apply_noise(
                preprocessed_sample, noise), verbose=0)[0][random_class])

        return noise.numpy()

    def _get_target_vector(self,
                           predictions: Union[np.ndarray, torch.Tensor, tf.Tensor],
                           true_class: int,
                           strategy: Literal['minimize', 'spread', 'uniform', 'random']
                           ) -> Union[np.ndarray, torch.Tensor, tf.Tensor]:
        num_classes = predictions.shape[1]

        if strategy == 'minimize':
            target_vector = predictions[1].copy()
            target_vector[true_class] = 0
            target_vector = target_vector / target_vector.sum()
        elif strategy == 'spread':
            target_vector = np.ones(num_classes) / (num_classes - 1)
            target_vector[true_class] = 1e-6
            target_vector /= target_vector.sum()
        elif strategy == 'uniform':
            target_vector = np.ones(num_classes) / num_classes
        elif strategy == 'random':
            random_class = random.choice([i for i in range(num_classes) if i != true_class])
            target_vector = np.zeros(shape=(num_classes, ))
            target_vector[random_class] = 1

        target_vector = np.expand_dims(target_vector, axis=0)
        target_vector = self.tensor_ops.to_tensor(target_vector)

        if predictions.shape != target_vector.shape:
            raise ValueError(
                "target_vector must be the same size as the outputs.")

        return target_vector
