from abc import ABC
import sys
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from adversarial_lab.callbacks import *
from adversarial_lab.core import ALModel
from adversarial_lab.attacker import AttackerBase
from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.analytics import AdversarialAnalytics
from adversarial_lab.core.losses import Loss, LossRegistry
from adversarial_lab.core.constraints import PostOptimizationConstraint
from adversarial_lab.core.optimizers import Optimizer, OptimizerRegistry
from adversarial_lab.core.preprocessing import NoPreprocessing, Preprocessing
from adversarial_lab.core.noise_generators import AdditiveNoiseGenerator, NoiseGenerator

from typing import Union, List, Optional, Literal, Callable


class BlackBoxAttack(AttackerBase):
    """
    Base class for black-box adversarial attacks using flexible update and gradient strategies.
    """

    def __init__(self,
                 predict_fn: Callable,
                 optimizer: Union[str, Optimizer],
                 loss: Optional[Union[str, Loss]] = None,
                 noise_generator: Optional[object] = None,
                 gradient_estimator: Optional[object] = None,
                 constraints: Optional[List[PostOptimizationConstraint]] = None,
                 analytics: Optional[AdversarialAnalytics] = None,
                 callbacks: Optional[List[Callback]] = None,
                 verbose: int = 1,
                 max_queries: int = 10000,
                 *args,
                 **kwargs) -> None:
        """
        Initializes the BlackBoxAttack with prediction function, update rule, gradient estimation, and other utilities.

        Args:
            predict_fn (callable): Function that takes a sample and returns either logits or labels.
            noise_generator (object): Update rule for adversarial example generation.
            gradient_estimation (object, optional): Module to estimate gradients if available.
            analytics (AdversarialAnalytics, optional): Analytics module for tracking metrics.
            callbacks (List[Callback], optional): List of callbacks to apply.
            verbose (int): Verbosity level.
            max_queries (int): Maximum allowed queries during attack.
        """
        super().__init__(model=predict_fn,
                         optimizer=optimizer,
                         loss=loss,
                         noise_generator=noise_generator,
                         constraints=constraints,
                         analytics=analytics,
                         callbacks=callbacks,
                         gradient_estimator=gradient_estimator,
                         verbose=verbose,
                         *args,
                         **kwargs)
        self.model.set_max_queries(max_queries)

    def attack(self, epochs, *args, **kwargs):
        super().attack(epochs, *args, **kwargs)
        self.model.reset_query_count()