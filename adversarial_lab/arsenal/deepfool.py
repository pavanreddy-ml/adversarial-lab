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
                        preds,
                        target_vector
                        ) -> None:
        noise = noise_meta[0]
        assert preds is not None and target_vector is not None

        f_0 = tf.tensordot(target_vector, jacobian, axes=1)  # shape: input_shape
        f_k_diff = tf.reduce_sum((preds - target_vector))
        w_k = tf.tensordot(tf.ones_like(target_vector), jacobian, axes=1) - f_0
        norm_w_k = tf.norm(tf.reshape(w_k, [-1]), ord=2)

        if norm_w_k == 0:
            return

        r_k = tf.abs(f_k_diff) / norm_w_k * (w_k / norm_w_k)
        noise.assign_add(r_k)


class ProjectedGradientDescentAttack(AttacksBase):
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
            noise_generator=DeepFoolNoiseGeenerator(),
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
