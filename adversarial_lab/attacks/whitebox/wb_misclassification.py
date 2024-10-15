from typing import Dict, Union, Literal

import random
import numpy as np
from tqdm import tqdm

import torch
import tensorflow as tf
from torch.nn import Module as TorchModel
from tensorflow.keras.models import Model as TFModel

from adversarial_lab.core.losses.loss_base import Loss
from adversarial_lab.core.preprocessing import Preprocessing
from adversarial_lab.core.noise_generators import NoiseGenerator
from adversarial_lab.core.optimizers.optimizer_base import Optimizer

from . import WhiteBoxAttack
from adversarial_lab.exceptions import VectorDimensionsError


class WhiteBoxMisclassification(WhiteBoxAttack):
    def __init__(self,
                 model: Union[TorchModel, TFModel],
                 loss: str | Loss,
                 optimizer: str | Optimizer,
                 noise_generator: NoiseGenerator = None,
                 preprocessing: Preprocessing = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(model, loss, optimizer,
                         noise_generator, preprocessing, *args, **kwargs)

    def attack(self,
               sample: Union[np.ndarray, torch.Tensor, tf.Tensor],
               target_class: int = None,
               target_vector: Union[np.ndarray, torch.Tensor, tf.Tensor] = None,
               strategy: Literal['spread', 'uniform', 'random'] = "random",
               epochs=10,
               *args,
               **kwargs
               ) -> np.ndarray:
        super().attack()

        # Vesbose is not used in this version. It will be used in future versions.
        verbose = kwargs.get("verbose", 1)

        # Future Versions must handle both pre and post preprocessing noise. The preprocessing function must be differentiable
        # In order for pre preprocessing noise.
        preprocessed_sample = self.preprocessing.preprocess(sample)
        noise = self.noise_generator.generate(preprocessed_sample)
        true_class = np.argmax(self.model.predict(preprocessed_sample), axis=1)[0]
        predictions = self.model.predict(self.noise_generator.apply_noise(preprocessed_sample, noise), verbose=0)

        if target_class and target_vector:
            raise ValueError("target_class and target_vector cannot be used together.")

        if target_class:
            if target_class >= self.model_info["output_shape"][1]:
                raise ValueError("target_class exceeds the dimension of the outputs.")
            target_vector = np.zeros(shape=(self.model_info["output_shape"][1], ))
            target_vector[target_class] = 1

        if target_vector:
            target_vector = self.tensor_ops.to_tensor(target_vector)
            if len(target_vector) != self.model_info["output_shape"][1]:
                raise VectorDimensionsError("target_vector must be the same size outputs.")
        else: 
            if strategy not in ['spread', 'uniform', 'random']:
                raise ValueError("Invalid value for strategy. It must be 'spread', 'uniform', 'random'.")
            target_vector = self._get_target_vector(predictions, true_class, strategy)


        for _ in range(epochs):
            gradients = self.get_grads.calculate(self.model, preprocessed_sample, noise, self.noise_generator, target_vector)
            self.noise_generator.apply_gradients(noise, gradients, self.optimizer)
            if self.noise_generator.use_constraints:
                noise = self.noise_generator.apply_constraints(noise)

            # Dev Only. Remove in Release
            print(self.model.predict(self.noise_generator.apply_noise(
                preprocessed_sample, noise), verbose=0)[0][true_class])

        return noise.numpy()

    def _get_target_vector(self,
                           predictions: Union[np.ndarray, torch.Tensor, tf.Tensor],
                           true_class: int,
                           strategy: Literal['minimize', 'spread', 'uniform', 'random']
                           ) -> Union[np.ndarray, torch.Tensor, tf.Tensor]:
        num_classes = predictions.shape[1]

        if strategy == 'spread':
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
            raise VectorDimensionsError("target_vector must be the same size as the outputs.")

        return target_vector
