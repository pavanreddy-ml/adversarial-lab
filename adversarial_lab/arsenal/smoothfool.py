from .attacks_base import AttacksBase

from adversarial_lab.attacker.whitebox import WhiteBoxMisclassification
from adversarial_lab.core.noise_generators import AdditiveNoiseGenerator
from adversarial_lab.core.losses import CategoricalCrossEntropy, BinaryCrossEntropy
from adversarial_lab.core.optimizers import PGD
from adversarial_lab.core.constraints import POClip
import numpy as np


class SmoothFoolNoiseGenerator(AdditiveNoiseGenerator):
    def __init__(self, lambda_tv=0.01, **kwargs):
        super().__init__(requires_jacobian=True, **kwargs)
        self.lambda_tv = lambda_tv

    def _tv_grad(self, x, epsilon=1e-8):
        arr = x.numpy()
        dx = np.roll(arr, -1, axis=-1) - arr
        dy = np.roll(arr, -1, axis=-2) - arr

        dx_norm = np.sqrt(dx ** 2 + epsilon)
        dy_norm = np.sqrt(dy ** 2 + epsilon)

        dx_grad = dx / dx_norm
        dy_grad = dy / dy_norm

        dx_back = dx_grad - np.roll(dx_grad, 1, axis=-1)
        dy_back = dy_grad - np.roll(dy_grad, 1, axis=-2)

        return self.tensor_ops.tensor(dx_back + dy_back)

    def apply_gradients(self,
                        noise_meta,
                        optimizer,
                        grads,
                        jacobian,
                        predictions,
                        target_vector
                        ) -> None:
        noise = noise_meta[0]

        jacobian_tensor = self.tensor_ops.remove_batch_dim(jacobian[0], axis=0)
        predictions = self.tensor_ops.remove_batch_dim(predictions, axis=0)
        target_vector = self.tensor_ops.remove_batch_dim(target_vector, axis=0)

        J_pred = self.tensor_ops.tensordot(predictions, jacobian_tensor, axes=[[0], [0]])
        J_target = self.tensor_ops.tensordot(target_vector, jacobian_tensor, axes=[[0], [0]])

        w_k = J_target - J_pred
        f_k_diff = self.tensor_ops.tensordot((target_vector - predictions), predictions, axes=1)

        w_k_flat = self.tensor_ops.reshape(w_k, [-1])
        norm_w_k = self.tensor_ops.norm(w_k_flat, p=2)

        if norm_w_k == 0:
            return

        r_k = self.tensor_ops.abs(f_k_diff) / norm_w_k * (w_k / norm_w_k)
        r_k_smoothed = r_k - self.lambda_tv * self._tv_grad(r_k)
        noise.assign(noise + r_k_smoothed)


class SmoothFoolAttack(AttacksBase):
    def __init__(self,
                 model,
                 preprocessing_fn,
                 learning_rate=0.01,
                 epsilon=0.1,
                 lambda_tv=0.01,
                 binary=False,
                 verbose=2,
                 *args,
                 **kwargs):
        self.attacker = WhiteBoxMisclassification(
            model=model,
            loss=CategoricalCrossEntropy() if not binary else BinaryCrossEntropy(),
            optimizer=PGD(learning_rate=learning_rate),
            noise_generator=SmoothFoolNoiseGenerator(lambda_tv=lambda_tv),
            preprocessing=preprocessing_fn,
            constraints=[POClip(min=-epsilon, max=epsilon)],
            verbose=verbose,
            *args,
            **kwargs
        )

    def attack(self,
               sample,
               target_class,
               target_vector,
               epochs=10,
               *args,
               **kwargs):
        return self.attacker.attack(sample=sample,
                                    target_class=target_class,
                                    target_vector=target_vector,
                                    epochs=epochs,
                                    *args,
                                    **kwargs)
