from abc import ABC, abstractmethod

import warnings
import traceback

from adversarial_lab.core.penalties import Penalty
from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.types import LossType, TensorType

from typing import Literal, List, Optional


class Loss():
    """
    Base class for all loss functions.
    """
    def __init__(self, 
                 penalties: Optional[List[Penalty]] = None,
                 from_logits: bool = False
                 ) -> None:
        """
        Initialize the loss function.

        Parameters:
            penalties (Optional[List[Penalty]]): A list of penalty functions to be 
                applied to the loss value. If None, no penalties are applied.
            from_logits (bool): Whether the loss is computed from prediction or logits.

        Raises:
            TypeError: If `penalties` is not a list of `Penalty` instances.
        
        Notes:
            - The `penalties` list allows for regularization or additional constraints 
              to be imposed on the loss function.
            - If penalties are provided, each must be an instance of `Penalty`.
        """
        self.value = None
        penalties = penalties if penalties is not None else []
        self.from_logits = from_logits

        self.warned = False

        if not all(isinstance(penalty, Penalty) for penalty in penalties):
            raise TypeError("penalties must be a list of Penalty instances")
        
        self.penalties = penalties

    @abstractmethod
    def calculate(self, 
                  target: TensorType, 
                  predictions: TensorType, 
                  logits: TensorType, 
                  *args, 
                  **kwargs
                  ) -> LossType:
        """
        Calculates the loss value.
        
        Args:
            target (TensorType): The target tensor.
            predictions (TensorType): The predictions tensor.
            logits (TensorType): The logits tensor.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            LossType: The computed loss value.
            
        Notes:
            - `logits` refers to raw model outputs before applying softmax or other 
              activation functions."""
        pass
    
    def set_value(self, 
                  value: LossType
                  ) -> None:
        """
        Set the loss value for tracking.

        Parameters:
            value (LossType): The computed loss value to be stored.

        Notes:
            - If an error occurs while setting the value, a warning is issued, and 
              `self.value` is set to `None`. The warning is issued only once.
        """
        try:
            if self.framework == "torch":
                self.value = value.item()
            elif self.framework == "tf":
                self.value = float(value.numpy())
        except Exception as e:
            if not self.warned:
                self.warned = True
                warnings.warn(f"Error while setting value: {e}. Traceback: {traceback.format_exc()}")

            self.value = None

    def set_framework(self, 
                      framework: Literal["tf", "torch"]
                      ) -> None:
        """
        Set the framework for the constraint.

        Args:
            framework: The framework to set, tf or torch.

        Raises:
            ValueError: If the framework is not 'tf' or 'torch'.
        """
        if framework not in ["tf", "torch"]:
            raise ValueError("framework must be either 'tf' or 'torch'")
        self.framework = framework
        self.tensor_ops = TensorOps(framework)

    def __repr__(self) -> str:
        """
        Return the string representation of the class.

        Returns:
            str: The name of the loss function class.
        """
        return self.__class__.__name__
