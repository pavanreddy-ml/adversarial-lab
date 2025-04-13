from .attacks_base import AttacksBase

from adversarial_lab.attacks.whitebox import WhiteBoxMisclassification
from adversarial_lab.core.noise_generators import AdditiveNoiseGenerator
from adversarial_lab.core.losses import CategoricalCrossEntropy, BinaryCrossEntropy
from adversarial_lab.core.optimizers import PGD
from adversarial_lab.core.constraints import POClip


class ProjectedGradientDescentAttack(AttacksBase):
    def __init__(self, 
                 model, 
                 preprocessing_fn, 
                 learning_rate= 0.01, 
                 epsilon=0.1, 
                 binary=False, 
                 verbose=2,
                 *args, 
                 **kwargs):
        self.attacker = WhiteBoxMisclassification(
            model=model,
            loss=CategoricalCrossEntropy() if not binary else BinaryCrossEntropy(),
            optimizer=PGD(learning_rate=learning_rate),
            noise_generator=AdditiveNoiseGenerator(),
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