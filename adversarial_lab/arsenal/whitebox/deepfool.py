from .attacks_base import AttacksBase

from adversarial_lab.attacker.whitebox import WhiteBoxMisclassification
from adversarial_lab.core.noise_generators import AdditiveNoiseGenerator
from adversarial_lab.core.losses import CategoricalCrossEntropy, BinaryCrossEntropy
from adversarial_lab.core.optimizers import PGD
from adversarial_lab.core.constraints import POClip

from adversarial_lab.core.types import ModelType, TensorType
from typing import Callable, Optional, Union, Literal


class DeepFoolNoiseGeenerator(AdditiveNoiseGenerator):
    def update(self,
               noise_meta,
               optimizer,
               grads,
               jacobian,
               predictions,
               target_vector,
               true_class,
               *args,
               **kwargs
               ) -> None:
        noise = noise_meta[0]

        jacobian_tensor = self.tensor_ops.remove_batch_dim(jacobian[0], axis=0)
        predictions = self.tensor_ops.remove_batch_dim(predictions, axis=0)
        target_vector = self.tensor_ops.remove_batch_dim(target_vector, axis=0)

        J_pred = self.tensor_ops.tensordot(
            predictions, jacobian_tensor, axes=[[0], [0]])
        J_target = self.tensor_ops.tensordot(
            target_vector, jacobian_tensor, axes=[[0], [0]])

        w_k = J_target - J_pred
        f_k_diff = self.tensor_ops.tensordot(
            (target_vector - predictions), predictions, axes=1)

        w_k_flat = self.tensor_ops.reshape(w_k, [-1])
        norm_w_k = self.tensor_ops.norm(w_k_flat, p=2)

        if norm_w_k == 0:
            return

        r_k = self.tensor_ops.abs(f_k_diff) / norm_w_k * (w_k / norm_w_k)
        noise.assign(noise + r_k)


class DeepFoolAttack(AttacksBase):
    def __init__(self,
                 model,
                 preprocessing_fn,
                 epsilon=0.1,
                 binary=False,
                 verbose=2,
                 *args,
                 **kwargs):
        self.attacker = WhiteBoxMisclassification(
            model=model,
            loss=CategoricalCrossEntropy() if not binary else BinaryCrossEntropy(),
            optimizer=PGD(learning_rate=0.1),
            noise_generator=DeepFoolNoiseGeenerator(requires_jacobian=True),
            preprocessing=preprocessing_fn,
            constraints=[POClip(min=-epsilon, max=epsilon)],
            verbose=verbose,
            *args,
            **kwargs
        )

    def attack(self,
               sample: TensorType,
               target_class: int = None,
               target_vector: TensorType = None,
               strategy: Literal['spread', 'uniform', 'random'] = "random",
               binary_threshold: float = 0.5,
               on_original: bool = False,
               epochs: int = 10,
               addn_analytics_fields: dict | None = None,
               *args,
               **kwargs):
        return self.attacker.attack(sample=sample,
                                    target_class=target_class,
                                    target_vector=target_vector,
                                    strategy=strategy,
                                    binary_threshold=binary_threshold,
                                    epochs=epochs,
                                    on_original=on_original,
                                    addn_analytics_fields=addn_analytics_fields,
                                    *args,
                                    **kwargs)
