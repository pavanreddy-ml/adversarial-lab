from typing import Dict, Union, Literal

import random
import numpy as np

import torch
import tensorflow as tf
from torch.nn import Module as TorchModel
from tensorflow.keras.models import Model as TFModel

from adversarial_lab.core.losses import Loss
from adversarial_lab.core.optimizers import Optimizer
from adversarial_lab.core.preprocessing import Preprocessing
from adversarial_lab.core.noise_generators import NoiseGenerator

from . import WhiteBoxAttack
from adversarial_lab.exceptions import VectorDimensionsError, IndifferentiabilityError


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
               binary_threshold: float = 0.5,
               epochs=10,
               *args,
               **kwargs
               ) -> np.ndarray:
        verbose = kwargs.get("verbose", 1)
        super().attack(epochs, *args, **kwargs)

        # Future Versions must handle both pre and post preprocessing noise. The preprocessing function must be differentiable
        # In order for pre preprocessing noise.
        preprocessed_sample = self.preprocessing.preprocess(sample)
        noise = self.noise_generator.generate(preprocessed_sample)
        predictions = self.model.predict(self.noise_generator.apply_noise(preprocessed_sample, noise), verbose=0) # Testing if noise can be pplied to the preprocessed image

        if self.model_info["output_shape"][1] == 1:
            true_class = (self.model.predict(preprocessed_sample) >= binary_threshold).astype(int)[0]
            target_class = 1 - true_class if not target_vector else None
        else:
            true_class = np.argmax(self.model.predict(preprocessed_sample), axis=1)[0]

        if target_class is not None and target_vector is not None:
            raise ValueError("target_class and target_vector cannot be used together.")

        if target_class:
            if target_class >= self.model_info["output_shape"][1]:
                raise ValueError("target_class exceeds the dimension of the outputs.")
            target_vector = np.zeros(shape=(self.model_info["output_shape"][1], ))
            target_vector[target_class] = 1

        if target_vector is not None:
            target_vector = self.tensor_ops.to_tensor(target_vector)
            if len(target_vector) != self.model_info["output_shape"][1]:
                raise VectorDimensionsError("target_vector must be the same size outputs.")
        else: 
            if strategy not in ['spread', 'uniform', 'random']:
                raise ValueError("Invalid value for strategy. It must be 'spread', 'uniform', 'random'.")
            target_vector = self._get_target_vector(predictions, true_class, strategy)
        
        for _ in range(epochs):
            self.noise_generator
            gradients, loss = self.get_grads.calculate(self.model, preprocessed_sample, noise, self.noise_generator, target_vector)
            if gradients is None:
                raise IndifferentiabilityError()

            self.noise_generator.apply_gradients(noise, gradients, self.optimizer)
            noise = self.noise_generator.apply_constraints(noise)

            # Stats
            predictions = self.model.predict(self.noise_generator.apply_noise(preprocessed_sample, noise), verbose=0)
            true_class = np.argmax(predictions, axis=1)[0]
            true_class_confidence = predictions[0][true_class]
            self.progress_bar.update(1)
            if verbose >= 2:
                self.progress_bar.set_postfix({'Loss': loss, 'Prediction': true_class, 'Prediction Confidence': true_class_confidence})

        return noise.numpy()

    def _get_target_vector(self,
                           predictions: Union[np.ndarray, torch.Tensor, tf.Tensor],
                           true_class: int,
                           strategy: Literal['spread', 'uniform', 'random']
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


