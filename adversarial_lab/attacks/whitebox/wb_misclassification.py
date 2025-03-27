import random
import numpy as np

import warnings

from . import WhiteBoxAttack
from adversarial_lab.core import ALModel
from adversarial_lab.core.losses import Loss
from adversarial_lab.core.optimizers import Optimizer
from adversarial_lab.analytics import AdversarialAnalytics
from adversarial_lab.exceptions import VectorDimensionsError
from adversarial_lab.core.preprocessing import Preprocessing
from adversarial_lab.core.noise_generators import NoiseGenerator
from adversarial_lab.core.constraints import PostOptimizationConstraint

from typing import Optional, Union, Literal
from adversarial_lab.core.types import TensorType, ModelType


class WhiteBoxMisclassification(WhiteBoxAttack):
    """
    WhiteBoxMisclassification is a white-box attack that generates adversarial examples by applying perturbations to the input sample.
    The attack aims to misclassify the input sample into a target class or target vector.
    It supports various strategies for generating the target vector and can handle binary and multi-class classification models.
    The attack uses a noise generator to create perturbations and an optimizer to update the noise during the attack process.
    """

    def __init__(self,
                 model: Union[ALModel, ModelType],
                 optimizer: Union[str, Optimizer],
                 loss: Optional[Union[str, Loss]] = None,
                 noise_generator: Optional[NoiseGenerator] = None,
                 preprocessing: Optional[Preprocessing] = None,
                 constraints: Optional[PostOptimizationConstraint] = None,
                 analytics: Optional[AdversarialAnalytics] = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(model=model,
                         loss=loss,
                         optimizer=optimizer,
                         noise_generator=noise_generator,
                         preprocessing=preprocessing,
                         constraints=constraints,
                         analytics=analytics,
                         *args,
                         **kwargs)

    def attack(self,
               sample: Union[np.ndarray, TensorType],
               target_class: int = None,
               target_vector: Union[np.ndarray, TensorType] = None,
               strategy: Literal['spread', 'uniform', 'random'] = "random",
               binary_threshold: float = 0.5,
               epochs=10,
               addn_analytics_fields: Optional[dict] = None,
               *args,
               **kwargs
               ) -> np.ndarray:
        verbose = kwargs.get("verbose", 1)
        addn_analytics_fields = addn_analytics_fields or {}
        super().attack(epochs, addn_analytics_fields, *args, **kwargs)

        preprocessed_sample, predictions, noise_meta, target_vector = self._initialize_attack(
            sample=sample,
            target_class=target_class,
            target_vector=target_vector,
            strategy=strategy,
            binary_threshold=binary_threshold
        )

        # Initial stats
        self.analytics.update_post_epoch_values(loss=self.loss,
                                                raw_image=sample,
                                                preprocessed_image=self.tensor_ops.remove_batch_dim(
                                                    preprocessed_sample).numpy(),
                                                noise_preprocessed_image=self.noise_generator.get_noise(
                                                    noise_meta),
                                                predictions=self.tensor_ops.remove_batch_dim(
                                                    predictions).numpy(),
                                                **addn_analytics_fields
                                                )
        self.analytics.write(epoch_num=0)

        for epoch in range(epochs):
            grads, jacobian = self.model.calculate_gradients(
                sample=preprocessed_sample,
                noise=noise_meta,
                construct_perturbation_fn=self.noise_generator.construct_perturbation,
                target_vector=target_vector,
                loss=self.loss)

            self.noise_generator.apply_gradients(
                noise_meta=noise_meta,
                optimizer=self.optimizer,
                grads=grads,
                jacobian=jacobian
            )

            self._apply_constrains(noise_meta)

            # Stats
            predictions = self.model.predict(preprocessed_sample + self.noise_generator.construct_perturbation(noise_meta))
            predictions = self.tensor_ops.remove_batch_dim(predictions)
            predicted_class = np.argmax(predictions)
            predicted_class_confidence = predictions[predicted_class].numpy()
            self.progress_bar.update(1)
            if verbose >= 2:
                self.progress_bar.set_postfix(
                    {'Loss': self.loss.value, 'Prediction': predicted_class, 'Prediction Confidence': predicted_class_confidence})

            self.analytics.update_post_epoch_values(loss=self.loss,
                                                    raw_image=sample,
                                                    preprocessed_image=self.tensor_ops.remove_batch_dim(
                                                        preprocessed_sample).numpy(),
                                                    noise_preprocessed_image=self.noise_generator.get_noise(
                                                        noise_meta),
                                                    predictions=self.tensor_ops.remove_batch_dim(
                                                        predictions).numpy(),
                                                    **addn_analytics_fields
                                                    )
            self.analytics.write(epoch_num=epoch)

        self.analytics.update_post_attack_values(loss=self.loss,
                                                 raw_image=sample,
                                                 preprocessed_image=self.tensor_ops.remove_batch_dim(
                                                     preprocessed_sample).numpy(),
                                                 noise_preprocessed_image=self.noise_generator.get_noise(
                                                     noise_meta),
                                                 predictions=self.tensor_ops.remove_batch_dim(
                                                     predictions).numpy(),
                                                **addn_analytics_fields
                                                 )
        self.analytics.write(epoch_num=9999999)

        return self.noise_generator.get_noise(noise_meta), noise_meta

    def _initialize_attack(self,
                           sample: Union[np.ndarray, TensorType],
                           target_class: int = None,
                           target_vector: Union[np.ndarray, TensorType] = None,
                           strategy: Literal['spread',
                                             'uniform', 'random'] = "random",
                           binary_threshold: float = 0.5,) -> None:
        # Future Versions must handle both pre and post preprocessing noise. The preprocessing function must be differentiable
        # In order for pre preprocessing noise.
        preprocessed_sample = self.preprocessing.preprocess(sample)

        if not self.tensor_ops.has_batch_dim(preprocessed_sample):
            has_batch_dim = False
            preprocessed_sample = self.tensor_ops.add_batch_dim(
                preprocessed_sample)
        else:
            has_batch_dim = True

        noise_meta = self.noise_generator.generate_noise_meta(
            preprocessed_sample)
        predictions = self.model.predict(preprocessed_sample + self.noise_generator.construct_perturbation(noise_meta))  # Testing if noise can be applied to the preprocessed image
        predictions = self.tensor_ops.add_batch_dim(predictions)

        if self.model.model_info["output_shape"][1] == 1:
            if target_class:
                warnings.warn(
                    "target_class are automatically set for binary classification. Ignoring provided target_class.")
            true_class = (predictions >= binary_threshold).astype(int)[0]
            target_class = 1 - true_class if not target_vector else None
        else:
            true_class = np.argmax(predictions, axis=1)[0]

        if target_class is not None and target_vector is not None:
            raise ValueError(
                "target_class and target_vector cannot be used together.")

        if target_class:
            if target_class >= self.model.model_info["output_shape"][1]:
                raise ValueError(
                    "target_class exceeds the dimension of the outputs.")
            target_vector = np.zeros(
                shape=(self.model.model_info["output_shape"][1], ))
            target_vector[target_class] = 1

        if target_vector is not None:
            target_vector = self.tensor_ops.tensor(target_vector)
            if len(target_vector) != self.model.model_info["output_shape"][1]:
                raise VectorDimensionsError(
                    "target_vector must be the same size outputs.")
        else:
            if strategy not in ['spread', 'uniform', 'random']:
                raise ValueError(
                    "Invalid value for strategy. It must be 'spread', 'uniform', 'random'.")
            target_vector = self._get_target_vector(
                predictions, true_class, strategy)

        if has_batch_dim:
            target_vector = self.tensor_ops.add_batch_dim(target_vector)
        else:
            preprocessed_sample = self.tensor_ops.remove_batch_dim(
                preprocessed_sample)
            target_vector = self.tensor_ops.remove_batch_dim(target_vector)

        return preprocessed_sample, predictions, noise_meta, target_vector

    def _get_target_vector(self,
                           predictions: Union[np.ndarray, TensorType],
                           true_class: int,
                           strategy: Literal['spread', 'uniform', 'random']
                           ) -> Union[np.ndarray, TensorType]:
        if self.tensor_ops.has_batch_dim(predictions):
            num_classes = predictions.shape[0]
        else:
            num_classes = predictions.shape[1]

        if strategy == 'spread':
            target_vector = np.ones(num_classes) / (num_classes - 1)
            target_vector[true_class] = 1e-6
            target_vector /= target_vector.sum()
        elif strategy == 'uniform':
            target_vector = np.ones(num_classes) / num_classes
        elif strategy == 'random':
            random_class = random.choice(
                [i for i in range(num_classes) if i != true_class])
            target_vector = np.zeros(shape=(num_classes, ))
            target_vector[random_class] = 1

        target_vector = self.tensor_ops.tensor(target_vector)

        if predictions.shape != target_vector.shape:
            raise VectorDimensionsError(
                "target_vector must be the same size as the outputs.")

        return target_vector
