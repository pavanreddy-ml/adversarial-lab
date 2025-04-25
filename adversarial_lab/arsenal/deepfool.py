from .attacks_base import AttacksBase

from adversarial_lab.attacks.whitebox import WhiteBoxMisclassification
from adversarial_lab.core.noise_generators import AdditiveNoiseGenerator
from adversarial_lab.core.losses import CategoricalCrossEntropy, BinaryCrossEntropy
from adversarial_lab.core.optimizers import PGD
from adversarial_lab.core.constraints import POClip


class DeepFoolNoiseGeenerator(AdditiveNoiseGenerator):
    def apply_gradients(self,
                        noise_meta,
                        optimizer,
                        grads,
                        jacobian,
                        predictions,
                        target_vector
                        ) -> None:
        noise = noise_meta[0]

        jacobian_tensor = self.tensor_ops.remove_batch_dim(jacobian[0], axis=0)  # shape: [num_classes, *input_shape]
        predictions = self.tensor_ops.remove_batch_dim(predictions, axis=0)  # shape: [num_classes]
        target_vector = self.tensor_ops.remove_batch_dim(target_vector, axis=0)  # shape: [num_classes]

        J_pred = self.tensor_ops.tensordot(predictions, jacobian_tensor, axes=[[0], [0]])  # shape: input_shape
        J_target = self.tensor_ops.tensordot(target_vector, jacobian_tensor, axes=[[0], [0]])  # shape: input_shape

        w_k = J_target - J_pred
        f_k_diff = self.tensor_ops.tensordot((target_vector - predictions), predictions, axes=1)  # scalar

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
                 learning_rate=0.01,
                 epsilon=0.1,
                 binary=False,
                 verbose=2,
                 *args,
                 **kwargs):
        self.attacker = WhiteBoxMisclassification(
            model=model,
            loss=CategoricalCrossEntropy() if not binary else BinaryCrossEntropy(),
            optimizer=PGD(learning_rate=learning_rate),
            noise_generator=DeepFoolNoiseGeenerator(requires_jacobian=True),
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
