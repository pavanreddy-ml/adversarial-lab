import random
import numpy as np

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
from adversarial_lab.core.types import TensorType


class WhiteBoxMisclassification(WhiteBoxAttack):
    """
    WhiteBoxMisclassification is a white-box attack that generates adversarial examples by applying perturbations to the input sample.
    The attack aims to misclassify the input sample into a target class or target vector.
    It supports various strategies for generating the target vector and can handle binary and multi-class classification models.
    The attack uses a noise generator to create perturbations and an optimizer to update the noise during the attack process.
    """

    def __init__(self,
                 model: ALModel,
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
               *args,
               **kwargs
               ) -> np.ndarray:
        verbose = kwargs.get("verbose", 1)
        super().attack(epochs, *args, **kwargs)

        # Future Versions must handle both pre and post preprocessing noise. The preprocessing function must be differentiable
        # In order for pre preprocessing noise.
        preprocessed_sample = self.preprocessing.preprocess(sample)
        noise_meta = self.noise_generator.generate_noise_meta(
            preprocessed_sample)
        predictions = self.model.predict(self.noise_generator.apply_noise(
            preprocessed_sample, noise_meta))  # Testing if noise can be pplied to the preprocessed image

        if self.model.model_info["output_shape"][1] == 1:
            true_class = (self.model.predict(preprocessed_sample)
                          >= binary_threshold).astype(int)[0]
            target_class = 1 - true_class if not target_vector else None
        else:
            true_class = np.argmax(self.model.predict(
                preprocessed_sample), axis=1)[0]

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

        self.analytics.update_post_epoch_values(epoch_num=0,
                                                  loss=self.loss,
                                                  raw_image=sample,
                                                  preprocessed_image=preprocessed_sample.numpy(),
                                                  noise_preprocessed_image=self.noise_generator.get_noise(
                                                      noise_meta),
                                                  predictions=predictions[0]
                                                  )
        self.analytics.write(epoch_num=0)

        for epoch in range(epochs):
            grads, logits_grad = self.model.calculate_gradients(
                preprocessed_sample, noise_meta, self.noise_generator.apply_noise, target_vector, self.loss)

            self.noise_generator.apply_gradients(
                noise_meta=noise_meta, grads=grads, logit_grads=logits_grad, optimizer=self.optimizer)
            self._apply_constrains(noise_meta)

            # Stats
            predictions = self.model.predict(
                self.noise_generator.apply_noise(preprocessed_sample, noise_meta))
            true_class = np.argmax(predictions, axis=1)[0]
            true_class_confidence = predictions[0][true_class].numpy()
            self.progress_bar.update(1)
            if verbose >= 2:
                self.progress_bar.set_postfix(
                    {'Loss': self.loss.value, 'Prediction': true_class, 'Prediction Confidence': true_class_confidence})

            self.analytics.update_post_epoch_values(epoch_num=epoch,
                                                    loss=self.loss,
                                                    raw_image=sample,
                                                    preprocessed_image=preprocessed_sample.numpy(),
                                                    noise_preprocessed_image=self.noise_generator.get_noise(
                                                        noise_meta),
                                                    predictions=predictions[0]
                                                    )
            self.analytics.write(epoch_num=epoch)

        self.analytics.update_post_attack_values(epoch_num=epoch,
                                                   loss=self.loss,
                                                   raw_image=sample,
                                                   preprocessed_image=preprocessed_sample.numpy(),
                                                   noise_preprocessed_image=self.noise_generator.get_noise(
                                                       noise_meta),
                                                   predictions=predictions[0]
                                                   )
        self.analytics.write(epoch_num=9999999)

        return noise_meta, self.noise_generator.get_noise(noise_meta)

    def _get_target_vector(self,
                           predictions: Union[np.ndarray, TensorType],
                           true_class: int,
                           strategy: Literal['spread', 'uniform', 'random']
                           ) -> Union[np.ndarray, TensorType]:
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

        target_vector = np.expand_dims(target_vector, axis=0)
        target_vector = self.tensor_ops.tensor(target_vector)

        if predictions.shape != target_vector.shape:
            raise VectorDimensionsError(
                "target_vector must be the same size as the outputs.")

        return target_vector
