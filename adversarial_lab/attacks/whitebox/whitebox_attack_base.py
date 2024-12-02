from abc import ABC, abstractmethod
from typing import Union, Any, Dict

import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
import tensorflow as tf
from torch.nn import Module as TorchModel
from tensorflow.keras.models import Model as TFModel


from adversarial_lab.core import ALModel
from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.analytics import AdversarialAnalytics
from adversarial_lab.core.losses import Loss, LossRegistry
from adversarial_lab.core.optimizers import Optimizer, OptimizerRegistry
from adversarial_lab.core.preprocessing import NoPreprocessing, Preprocessing
from adversarial_lab.core.noise_generators import AdditiveNoiseGenerator, NoiseGenerator

class WhiteBoxAttack(ABC):
    """
    Abstract class representing a white-box adversarial attack. Subclasses must implement specific
    attack methods.

    Parameters:
    ----------
    model : Union[TorchModel, TFModel]
        The machine learning model being attacked, either a PyTorch or TensorFlow model.
    loss : Union[str, Loss]
        Loss function used to optimize the attack. Can either be a string key for a registered loss
        or an instance of a `Loss` object.
    optimizer : Union[str, Optimizer]
        Optimizer used to update the noise in the adversarial attack. Can be either a string key for
        a registered optimizer or an instance of an `Optimizer` object.
    noise_generator : NoiseGenerator, optional
        A noise generator that creates perturbations for the adversarial attack. Default is 
        `AdditiveNoiseGenerator` if not provided.
    preprocessing : Preprocessing, optional
        Preprocessing pipeline applied to the input before generating adversarial noise. Default is
        `NoPreprocessing` if not provided.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """
    def __init__(self, 
                 model: ALModel,
                 loss: Union[str, Loss],
                 optimizer: Union[str, Optimizer],
                 noise_generator: NoiseGenerator = None,
                 preprocessing: Preprocessing = None,
                 analytics: AdversarialAnalytics = None,
                 *args,
                 **kwargs
                 ) -> None:
        """
        Initializes the white-box attack, including setting the framework (PyTorch or TensorFlow),
        and setting up the model, loss function, optimizer, noise generator, and preprocessing.
        """
        self.model = model
        self.framework: str = self.model.framework

        self._optimizer_arg = optimizer
        self._initialize_optimizer(self._optimizer_arg)
        self._initialize_loss(loss)
        self._initialize_noise_generator(noise_generator)
        self._initialize_preprocessing(preprocessing)

        if analytics is not None:
            if not isinstance(analytics, AdversarialAnalytics):
                raise ValueError("analytics must be an instance of AdversarialAnalytics")
            self.analytics = analytics
        else:
            self.analytics = AdversarialAnalytics(db=None, trackers=[], table_name=None)

        self.tensor_ops = TensorOps(framework=self.framework)
        
    def attack(self,
               epochs: int,
               *args,
               **kwargs
               ) -> np.ndarray:
        """
        Executes the adversarial attack for the specified number of epochs. Reinitialize the optimizer to track new variables.
        Sets up the progress bar for the attack.

        Parameters:
        ----------
        epochs : int
            The number of epochs to run the attack for.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments, including:
            - verbose (int): Verbosity level. 0 for no output, 1 for progress bar, 2 and higher for detailed text output.

        Returns:
        -------
        np.ndarray
            The adversarial noise generated during the attack.
        """
        self._initialize_optimizer(self._optimizer_arg)
        verbose = kwargs.get("verbose", 1)
        self.progress_bar = tqdm(total=epochs, desc="Attacking", leave=True, disable=(verbose == 0))

    def _initialize_optimizer(self, optimizer):
        """
        Initializes the optimizer for the attack based on the input. Can accept a string or an
        `Optimizer` instance.

        Parameters:
        ----------
        optimizer : Union[str, Optimizer]
            The optimizer to use for updating adversarial noise.

        Raises:
        -------
        TypeError
            If the optimizer argument is neither a string nor an instance of `Optimizer`.
        """
        optimizer_copy = deepcopy(optimizer)

        if isinstance(optimizer_copy, str):
            optimizer_class = OptimizerRegistry.get(optimizer_copy)
            self.optimizer = optimizer_class(framework=self.framework)
        elif isinstance(optimizer_copy, Optimizer):
            self.optimizer = optimizer_copy
        else:
            raise TypeError(f"Invalid type for optimizer: '{type(optimizer_copy)}'")

    def _initialize_loss(self, loss):
        """
        Initializes the loss function for the attack based on the input. Can accept a string or a
        `Loss` instance.

        Parameters:
        ----------
        loss : Union[str, Loss]
            The loss function to use for optimizing the attack.

        Raises:
        -------
        TypeError
            If the loss argument is neither a string nor an instance of `Loss`.
        """
        if isinstance(loss, str):
            loss_class = LossRegistry.get(loss)
            self.loss = loss_class(framework=self.framework)
        elif isinstance(loss, Loss):
            self.loss = loss
        else:
            raise TypeError(f"Invalid type for loss: '{type(loss)}'")
        
    def _initialize_noise_generator(self, noise_generator):
        """
        Initializes the noise generator for the attack. If no noise generator is provided,
        an `AdditiveNoiseGenerator` is used by default.

        Parameters:
        ----------
        noise_generator : NoiseGenerator, optional
            The noise generator to use for creating perturbations in the attack.

        Raises:
        -------
        TypeError
            If the noise generator is not an instance of `NoiseGenerator`.
        """
        if noise_generator is None:
            self.noise_generator = AdditiveNoiseGenerator(framework=self.framework, scale=(-0.05, 0.05), dist='uniform')
        elif isinstance(noise_generator, NoiseGenerator):
            self.noise_generator = noise_generator
        else:
            raise TypeError(f"Invalid type for noise_generator: '{type(noise_generator)}'")
        
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
            raise TypeError(f"Invalid type for preprocessing: '{type(preprocessing)}'")