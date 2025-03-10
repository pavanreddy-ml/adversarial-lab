from .loss_base import Loss

from .categorical_cross_entropy import CategoricalCrossEntropy
from .binary_cross_entropy import BinaryCrossEntropy
from .mean_absolute_error import MeanAbsoluteError
from .mean_squared_error import MeanSquaredError
from .dummy import DummyLoss

from .loss_registry import LossRegistry
from .loss_from_function import LossFromFunction


__all__ = [
    "Loss",
    "LossFromFunction",
    "LossRegistry",
    "CategoricalCrossEntropy",
    "BinaryCrossEntropy",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "DummyLoss"
]
