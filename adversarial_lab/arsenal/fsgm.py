from .attacks_base import AttacksBase

from adversarial_lab.attacks.whitebox import WhiteBoxMisclassification
from adversarial_lab.core.noise_generators import AdditiveNoiseGenerator
from adversarial_lab.core.losses import CategoricalCrossEntropy
from adversarial_lab.core.optimizers import PGD


class FastSignGradientMethodAttack(AttacksBase):
    def __init__(self,
                 model,
                 preprocessing_fn,
                 epsilon=0.1,
                 *args,
                 **kwargs
                 ):
        self.optimizer = PGD(learning_rate=epsilon)
        self.noise_generator = AdditiveNoiseGenerator()

        self.attacker = WhiteBoxMisclassification(
            model=model,
            loss=CategoricalCrossEntropy(),
            optimizer=self.optimizer,
            noise_generator=self.noise_generator,
            preprocessing=preprocessing_fn,
            *args,
            **kwargs
        )

    def attack(self,
               sample,
               target_class,
               target_vector,
               *args,
               **kwargs):
        return self.attacker.attack(sample=sample,
                                    target_class=target_class,
                                    target_vector=target_vector,
                                    epochs=1,
                                    *args,
                                    **kwargs)
