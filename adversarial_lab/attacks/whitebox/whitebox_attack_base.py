from abc import ABC

import numpy as np
from tqdm import tqdm
from copy import deepcopy

from adversarial_lab.core import ALModel
from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.analytics import AdversarialAnalytics
from adversarial_lab.core.losses import Loss, LossRegistry
from adversarial_lab.core.constraints import PostOptimizationConstraint
from adversarial_lab.core.optimizers import Optimizer, OptimizerRegistry
from adversarial_lab.core.preprocessing import NoPreprocessing, Preprocessing
from adversarial_lab.core.noise_generators import AdditiveNoiseGenerator, NoiseGenerator

from typing import Union, List, Optional


class WhiteBoxAttack(ABC):
    """
    Base class for white-box adversarial attack. Subclasses must implement specific attack methods.
    """

    def __init__(self,
                 model: ALModel,
                 optimizer: Union[str, Optimizer],
                 loss: Optional[Union[str, Loss]] = None,
                 noise_generator: Optional[NoiseGenerator] = None,
                 preprocessing: Optional[Preprocessing] = None,
                 constraints: Optional[List[PostOptimizationConstraint]] = None,
                 analytics: Optional[AdversarialAnalytics] = None,
                 *args,
                 **kwargs
                 ) -> None:
        """
        Initializes the WhiteBoxAttack with the given model, loss function, optimizer, noise generator,
        preprocessing, constraints, and analytics.

        Args:
            model (ALModel, torch.nn.Module, tf.keras.Model, tf.keras.Sequential): 
                The model to be attacked.
            loss (Union[str, Loss], optional): 
                The loss function to use for the attack. If a string, retrieves the loss from the LossRegistry.
            optimizer (Union[str, Optimizer]): 
                The optimizer to use for the attack. If a string, retrieves the optimizer from the OptimizerRegistry.
            noise_generator (NoiseGenerator, optional): 
                The noise generator to use for the attack. If None, `AdditiveNoiseGenerator` will be used.
            preprocessing (Preprocessing, optional): 
                The preprocessing method to use. If None, `NoPreprocessing` will be used.
            constraints (List[PostOptimizationConstraint], optional): 
                A list of constraints to apply on the noise after the attack, in order. If None, no constraints will be applied.
            analytics (AdversarialAnalytics, optional): 
                The analytics module for tracking attack metrics. If None, no metrics will be tracked.
            *args (tuple): 
                Additional positional arguments.
            **kwargs (dict): 
                Additional keyword arguments.

        Raises:
            TypeError: If:
                - `loss` is not an instance of `Loss` or a valid string.
                - `optimizer` is not an instance of `Optimizer` or a valid string.
                - `noise_generator` is not an instance of `NoiseGenerator`.
                - `preprocessing` is not an instance of `Preprocessing`.
                - `constraints` are not instances of `PostOptimizationConstraint`.
                - `analytics` is not an instance of `AdversarialAnalytics`.

        Notes:
                - `compute_jacobian` determines whether the Jacobian is computed. 
                This is set by `noise_generator` and defaults to `False`. 
                To enable, set `requires_jacobian=True` whilie initializing noise generator.
        """
        self.model = model if isinstance(model, ALModel) else ALModel(model)
        self.framework: str = self.model.framework

        self._optimizer_arg = optimizer
        self._initialize_optimizer(self._optimizer_arg)
        self._initialize_loss(loss)
        self._initialize_noise_generator(noise_generator)
        self._initialize_preprocessing(preprocessing)
        self._initialize_constraints(constraints)
        self._initialize_analytics(analytics)

        self.tensor_ops = TensorOps(framework=self.framework)

    def attack(self,
               epochs: int,
               *args,
               **kwargs
               ) -> np.ndarray:
        """
        Launch the attack process.

        Parameters:
            epochs (int): The number of epochs to run the attack.
            *args (tuple): Additional positional arguments.
            **kwargs (dict): Additional keyword arguments, including:
                - verbose (int): Verbosity level. 
                    - `0`: No output.
                    - `1`: Progress bar.
                    - `2+`: Detailed text output.

        Returns:
            np.ndarray: The adversarial noise generated during the attack.
        """
        self._initialize_optimizer(self._optimizer_arg)
        verbose = kwargs.get("verbose", 1)
        self.progress_bar = tqdm(
            total=epochs, desc="Attacking", leave=True, disable=(verbose == 0))

    def _initialize_optimizer(self, optimizer):
        """
        Initialize the optimizer for the attack.

        The optimizer can be specified as either a string or an `Optimizer` instance.

        Parameters:
            optimizer (Union[str, Optimizer]): The optimizer to use for updating adversarial noise.

        Raises:
            TypeError: If `optimizer` is neither a string nor an instance of `Optimizer`.

        Notes:
            - If `optimizer` is a string, it will be retrieved from the `OptimizerRegistry`.
            - If `optimizer` is an instance of `Optimizer`, it will be used directly.
            - The framework of the optimizer is set to match the model's framework.
            - The optimizer is deep-copied to avoid modifying the original instance. This will be used
            to reinitialize the optimizer for each attack to track the new Noise Tensor
        """

        optimizer_copy = deepcopy(optimizer)

        if isinstance(optimizer_copy, str):
            optimizer_class = OptimizerRegistry.get(optimizer_copy)
            self.optimizer = optimizer_class()
        elif isinstance(optimizer_copy, Optimizer):
            self.optimizer = optimizer_copy
        else:
            raise TypeError(
                f"Invalid type for optimizer: '{type(optimizer_copy)}'")

        self.optimizer.set_framework(self.framework)

    def _initialize_loss(self, loss):
        """
        Initialize the loss function for the attack.

        The loss function can be specified as either a string or a `Loss` instance. The framework
        for the loss function is set to match the model's framework.

        Parameters:
            loss (Union[str, Loss]): The loss function to use for the attack.

        Raises:
            TypeError: If `loss` is neither a string nor an instance of `Loss`.

        Notes:
            - If `loss` is a string, it will be retrieved from the `LossRegistry`.
            - If `loss` is an instance of `Loss`, it will be used directly.
            - The framework of the loss function is set to match the model's framework.
        """

        if isinstance(loss, (str, type(None))):
            loss_class = LossRegistry.get(loss)
            self.loss = loss_class()
        elif isinstance(loss, Loss):
            self.loss = loss
        else:
            raise TypeError(f"Invalid type for loss: '{type(loss)}'")

        self.loss.set_framework(self.framework)

    def _initialize_noise_generator(self, noise_generator):
        """
        Initialize the noise generator for the attack. If no noise generator is provided, 
        `AdditiveNoiseGenerator` is used by default.

        Parameters:
            noise_generator (NoiseGenerator, optional): The noise generator to use for the attack.

        Raises:
            TypeError: If `noise_generator` is not an instance of `NoiseGenerator`.
            ValueError: If `noise_generator` is not a valid type.

        Notes:
            - If `noise_generator` is None, an instance of `AdditiveNoiseGenerator` with a uniform 
            distribution and scale (-0.05, 0.05) is used by default.
            - If `noise_generator` is provided, it must be an instance of `NoiseGenerator`.
            - The framework of the noise generator is set to match the model's framework.
            - The model's `compute_jacobian` flag is set based on whether the noise generator 
            requires the Jacobian.
        """

        if noise_generator is None:
            self.noise_generator = AdditiveNoiseGenerator(
                scale=(-0.05, 0.05), dist='uniform')
        elif isinstance(noise_generator, NoiseGenerator):
            self.noise_generator = noise_generator
        else:
            raise TypeError(
                f"Invalid type for noise_generator: '{type(noise_generator)}'")
        self.noise_generator.set_framework(self.framework)
        self.model.set_compute_jacobian(self.noise_generator.requires_jacobian)

    def _initialize_preprocessing(self, preprocessing):
        """
        Initializes the preprocessing pipeline for the attack. If no preprocessing is provided,
        `NoPreprocessing` is used by default.

        Parameters:
        ----------
        preprocessing : Preprocessing, optional
            The preprocessing pipeline to apply before generating adversarial noise.

        Raises:
        -------
        TypeError
            If the preprocessing argument is not an instance of `Preprocessing`.
        """
        if preprocessing is None:
            self.preprocessing = NoPreprocessing()
        elif isinstance(preprocessing, Preprocessing):
            self.preprocessing = preprocessing
        else:
            raise TypeError(
                f"Invalid type for preprocessing: '{type(preprocessing)}'")

    def _initialize_constraints(self,
                                constraints: List[PostOptimizationConstraint]):
        """
        Initialize the constraints for the attack. If no constraints are provided, an empty list is used by default.

        Parameters:
            constraints (List[PostOptimizationConstraint], optional): 
                The constraints to apply to the adversarial noise after optimization.

        Raises:
            TypeError: If any element in `constraints` is not an instance of `PostOptimizationConstraint`.

        Notes:
            - If `constraints` is None, an empty list is used as the default.
            - Each constraint must be an instance of `PostOptimizationConstraint`.
            - The framework of each constraint is set to match the model's framework.
        """
        if constraints is None:
            constraints = []

        for contraint in constraints:
            if not isinstance(contraint, PostOptimizationConstraint):
                raise TypeError(
                    f"Invalid type for constraints: '{type(contraint)}'")

        for constraint in constraints:
            constraint.set_framework(self.framework)

        self.constraints = constraints

    def _initialize_analytics(self, analytics):
        """
        Initialize the analytics module for tracking attack metrics. If no analytics module is provided, 
        a default `AdversarialAnalytics` instance is created.

        Parameters:
            analytics (AdversarialAnalytics, optional): 
                The analytics module to use for tracking attack metrics.

        Raises:
            TypeError: If `analytics` is not an instance of `AdversarialAnalytics`.

        Notes:
            - If `analytics` is None, a default `AdversarialAnalytics` instance is initialized 
            with no database, no trackers, and no table name. Will not track any metrics.
        """
        if analytics is not None:
            if not isinstance(analytics, AdversarialAnalytics):
                raise TypeError(
                    "analytics must be an instance of AdversarialAnalytics")
            self.analytics = analytics
        else:
            self.analytics = AdversarialAnalytics(
                db=None, trackers=[], table_name=None)

    def _apply_constrains(self, noise):
        """
        Apply all constraints to the noise tensor.

        Parameters:
            noise (tf.Variable, torch.Tensor): The noise tensor to which constraints are applied.

        Notes:
            - Each constraint in `self.constraints` is applied in sequence.
        """
        for constraint in self.constraints:
            for n in noise:
                constraint.apply(n)
