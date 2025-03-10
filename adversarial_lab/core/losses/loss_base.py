from abc import ABC, abstractmethod
import warnings
import traceback

from adversarial_lab.core.penalties import Penalty
from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.types import LossType, TensorType

from typing import Literal, List, Optional


class Loss(ABC):
    """
    Base class for all loss functions.
    """

    def __init__(self, penalties: Optional[List[Penalty]] = None, from_logits: bool = False) -> None:
        """
        Initialize the loss function.

        Args:
            penalties: A list of penalty functions to be applied to the loss value. If None, no penalties are applied.
            from_logits: Whether the loss is computed from prediction or logits.

        Raises:
            TypeError: If `penalties` is not a list of `Penalty` instances.

        Note:
            The `penalties` list allows for regularization or additional constraints to be imposed on the loss function.
            If penalties are provided, each must be an instance of `Penalty`.
        """
        self.value = None
        self.from_logits = from_logits
        self.warned = False

        penalties = penalties if penalties is not None else []
        if not all(isinstance(penalty, Penalty) for penalty in penalties):
            raise TypeError("penalties must be a list of Penalty instances")

        self.penalties = penalties

    @abstractmethod
    def calculate(self, target: TensorType, predictions: TensorType, logits: TensorType, *args, **kwargs) -> LossType:
        """
        Calculates the loss value.

        Args:
            target: The target tensor.
            predictions: The predictions tensor.
            logits: The logits tensor.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.

        Note:
            If `from_logits=True`, `predictions` may be ignored, and `logits` are directly used.
        """
        pass

    def set_value(self, value: LossType) -> None:
        """
        Set the loss value for tracking.

        Args:
            value: The computed loss value to be stored.

        Note:
            If an error occurs while setting the value, a warning is issued, and `self.value` is set to `None`.
            The warning is issued only once.
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

    def set_framework(self, framework: Literal["tf", "torch"]) -> None:
        """
        Set the framework for the constraint.

        Args:
            framework: The framework to set, either 'tf' or 'torch'.

        Raises:
            ValueError: If the framework is not 'tf' or 'torch'.

        Note:
            Updates `self.tensor_ops` to match the selected framework.
        """
        if framework not in ["tf", "torch"]:
            raise ValueError("framework must be either 'tf' or 'torch'")
        self.framework = framework
        self.tensor_ops = TensorOps(framework)

    def __repr__(self) -> str:
        """
        Return the string representation of the class.

        Returns:
            The name of the loss function class.
        """
        return self.__class__.__name__
