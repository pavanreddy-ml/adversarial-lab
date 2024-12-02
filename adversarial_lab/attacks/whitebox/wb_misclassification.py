from typing import Dict, Union, Literal

import random
import numpy as np

import torch
import tensorflow as tf
from torch.nn import Module as TorchModel
from tensorflow.keras.models import Model as TFModel

from adversarial_lab.core import ALModel
from adversarial_lab.core.losses import Loss
from adversarial_lab.core.optimizers import Optimizer
from adversarial_lab.analytics import AdversarialAnalytics
from adversarial_lab.core.preprocessing import Preprocessing
from adversarial_lab.core.noise_generators import NoiseGenerator

from . import WhiteBoxAttack
from adversarial_lab.exceptions import VectorDimensionsError, IndifferentiabilityError


class WhiteBoxMisclassification(WhiteBoxAttack):
    """
    WhiteBoxMisclassification is a subclass of `WhiteBoxAttack` that performs white-box adversarial 
    attacks targeting misclassification. It supports both PyTorch and TensorFlow models.

    Parameters
    ----------
    model : Union[TorchModel, TFModel]
        The machine learning model to attack, either a PyTorch (`torch.nn.Module`) or TensorFlow (`tf.keras.Model`) model.
    loss : Union[str, Loss]
        The loss function to optimize the attack. It can either be a string that refers to a registered loss
        or an instance of a `Loss` object.
    optimizer : Union[str, Optimizer]
        The optimizer to use for updating the noise during the attack. It can either be a string referring to 
        a registered optimizer or an instance of `Optimizer`.
    noise_generator : NoiseGenerator, optional
        The noise generator used to create perturbations. If not provided, a default noise generator will be used.
    preprocessing : Preprocessing, optional
        The preprocessing pipeline to apply before generating adversarial noise. If not provided, no preprocessing will be applied.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.

    Methods
    -------
    attack(sample, target_class=None, target_vector=None, strategy='random', binary_threshold=0.5, epochs=10, *args, **kwargs)
        Executes the white-box adversarial attack by generating noise and applying it to the input sample.
    
    _get_target_vector(predictions, true_class, strategy)
        Generates a target vector for the attack based on the specified strategy (spread, uniform, or random).
    """
    def __init__(self,
                 model: ALModel,
                 loss: str | Loss,
                 optimizer: str | Optimizer,
                 noise_generator: NoiseGenerator = None,
                 preprocessing: Preprocessing = None,
                 analytics: AdversarialAnalytics = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(model=model, 
                         loss=loss, 
                         optimizer=optimizer,
                         noise_generator=noise_generator, 
                         preprocessing=preprocessing, 
                         analytics=analytics, 
                         *args, 
                         **kwargs)

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
        """
        Executes the white-box misclassification attack by applying perturbations to the input sample.

        Parameters
        ----------
        sample : Union[np.ndarray, torch.Tensor, tf.Tensor]
            The input sample to be attacked. This can be a NumPy array, PyTorch tensor, or TensorFlow tensor.
        target_class : int, optional
            The target class to misclassify the sample into. If provided, `target_vector` should not be provided.
        target_vector : Union[np.ndarray, torch.Tensor, tf.Tensor], optional
            A target vector that specifies the probability distribution over all output classes. If provided, `target_class` should not be provided.
        strategy : Literal['spread', 'uniform', 'random'], optional
            The strategy used to generate the target vector. Options are:
            - 'spread': The target is spread across all classes except the true class.
            - 'uniform': The target is uniformly distributed across all classes.
            - 'random': The target is set to a random class that is not the true class.
            The default is 'random'.
        binary_threshold : float, optional
            The threshold for binary classification models. If the model outputs a scalar, the prediction is thresholded at this value.
            The default is 0.5.
        epochs : int, optional
            The number of epochs (iterations) to run the attack for. The default is 10.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments, including:
            - verbose (int): Level of verbosity (0: no output, 1: progress bar, 2 or more: detailed information).

        Returns
        -------
        np.ndarray
            The adversarial noise generated during the attack.
        
        Raises
        ------
        ValueError
            If both `target_class` and `target_vector` are provided.
        VectorDimensionsError
            If `target_vector` has the wrong dimensions or does not match the output shape of the model.
        IndifferentiabilityError
            If the gradients for the attack cannot be computed.
        """
        verbose = kwargs.get("verbose", 1)
        super().attack(epochs, *args, **kwargs)

        # Future Versions must handle both pre and post preprocessing noise. The preprocessing function must be differentiable
        # In order for pre preprocessing noise.
        preprocessed_sample = self.preprocessing.preprocess(sample)
        noise_meta = self.noise_generator.generate_noise_meta(preprocessed_sample)
        predictions = self.model.predict(self.noise_generator.apply_noise(preprocessed_sample, noise_meta)) # Testing if noise can be pplied to the preprocessed image

        if self.model.model_info["output_shape"][1] == 1:
            true_class = (self.model.predict(preprocessed_sample) >= binary_threshold).astype(int)[0]
            target_class = 1 - true_class if not target_vector else None
        else:
            true_class = np.argmax(self.model.predict(preprocessed_sample), axis=1)[0]

        if target_class is not None and target_vector is not None:
            raise ValueError("target_class and target_vector cannot be used together.")

        if target_class:
            if target_class >= self.model.model_info["output_shape"][1]:
                raise ValueError("target_class exceeds the dimension of the outputs.")
            target_vector = np.zeros(shape=(self.model.model_info["output_shape"][1], ))
            target_vector[target_class] = 1

        if target_vector is not None:
            target_vector = self.tensor_ops.to_tensor(target_vector)
            if len(target_vector) != self.model.model_info["output_shape"][1]:
                raise VectorDimensionsError("target_vector must be the same size outputs.")
        else: 
            if strategy not in ['spread', 'uniform', 'random']:
                raise ValueError("Invalid value for strategy. It must be 'spread', 'uniform', 'random'.")
            target_vector = self._get_target_vector(predictions, true_class, strategy)

        self.analytics.update_post_epoch_values(epoch_num=0, 
                                                loss=self.loss, 
                                                raw_image=sample,
                                                preprocessed_image=preprocessed_sample,
                                                noise=self.noise_generator.get_noise(noise_meta),
                                                noised_image=None,
                                                noised_preprocessed_image=self.noise_generator.apply_noise(preprocessed_sample, noise_meta),
                                                predictions=predictions[0]
                                                    )
        self.analytics.write(epoch_num=0)
        
        for epoch in range(epochs):
            gradients, loss = self.model.calculate_gradients(preprocessed_sample, noise_meta, self.noise_generator, target_vector, self.loss)

            self.noise_generator.apply_gradients(noise_meta, gradients, self.optimizer)
            self.noise_generator.apply_constraints(noise_meta)

            # Stats
            predictions = self.model.predict(self.noise_generator.apply_noise(preprocessed_sample, noise_meta))
            true_class = np.argmax(predictions, axis=1)[0]
            true_class_confidence = predictions[0][true_class]
            self.progress_bar.update(1)
            if verbose >= 2:
                self.progress_bar.set_postfix({'Loss': loss, 'Prediction': true_class, 'Prediction Confidence': true_class_confidence})


            self.analytics.update_post_epoch_values(epoch_num=epoch, 
                                                    loss=self.loss, 
                                                    raw_image=sample,
                                                    preprocessed_image=preprocessed_sample,
                                                    noise=self.noise_generator.get_noise(noise_meta),
                                                    noised_image=None,
                                                    noised_preprocessed_image=self.noise_generator.apply_noise(preprocessed_sample, noise_meta),
                                                    predictions=predictions[0]
                                                    )
            self.analytics.write(epoch_num=epoch)
            
        self.analytics.update_post_training_values(epoch_num=epoch, 
                                                    loss=self.loss, 
                                                    raw_image=sample,
                                                    preprocessed_image=preprocessed_sample,
                                                    noise=self.noise_generator.get_noise(noise_meta),
                                                    noised_image=None,
                                                    noised_preprocessed_image=self.noise_generator.apply_noise(preprocessed_sample, noise_meta),
                                                    predictions=predictions[0]
                                                    )
        self.analytics.write(epoch_num=9999999)
        
        return noise_meta, self.noise_generator.get_noise(noise_meta)

    def _get_target_vector(self,
                           predictions: Union[np.ndarray, torch.Tensor, tf.Tensor],
                           true_class: int,
                           strategy: Literal['spread', 'uniform', 'random']
                           ) -> Union[np.ndarray, torch.Tensor, tf.Tensor]:
        """
        Generates a target vector for the adversarial attack based on the given strategy.

        Parameters
        ----------
        predictions : Union[np.ndarray, torch.Tensor, tf.Tensor]
            The model's output predictions for the input sample.
        true_class : int
            The index of the true class for the input sample.
        strategy : Literal['spread', 'uniform', 'random']
            The strategy for generating the target vector. Options are:
            - 'spread': The target is spread across all classes except the true class.
            - 'uniform': The target is uniformly distributed across all classes.
            - 'random': The target is set to a random class that is not the true class.

        Returns
        -------
        Union[np.ndarray, torch.Tensor, tf.Tensor]
            The generated target vector, in the same framework format as the input `predictions`.

        Raises
        ------
        VectorDimensionsError
            If the shape of the generated target vector does not match the shape of the model's predictions.
        """
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


