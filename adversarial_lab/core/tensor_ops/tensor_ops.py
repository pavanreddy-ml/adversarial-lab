from abc import ABCMeta
from typing import Literal, Union, TypeVar, Generic
import torch
import tensorflow as tf

from .tensor_tf import TensorOpsTF
from .tensor_torch import TensorOpsTorch

T = TypeVar('T', TensorOpsTorch, TensorOpsTF)


class TensorOpsMeta(ABCMeta):
    def __new__(cls, name: str, bases: tuple, dct: dict) -> 'TensorOpsMeta':
        cls.validate_methods_consistency()
        return super().__new__(cls, name, bases, dct)

    @staticmethod
    def validate_methods_consistency() -> None:
        torch_methods = set(dir(TensorOpsTorch))
        tf_methods = set(dir(TensorOpsTF))

        torch_methods = {m for m in torch_methods if not m.startswith("__")}
        tf_methods = {m for m in tf_methods if not m.startswith("__")}

        torch_only = torch_methods - tf_methods
        tf_only = tf_methods - torch_methods

        if torch_only or tf_only:
            error_message = "Method inconsistency found between TensorOpsTorch and TensorOpsTF.\n"

            if torch_only:
                error_message += f"Methods in TensorOpsTorch not in TensorOpsTF: {torch_only}\n"
            if tf_only:
                error_message += f"Methods in TensorOpsTF not in TensorOpsTorch: {tf_only}\n"

            raise NotImplementedError(
                error_message +
                "For consistency, please implement these methods in both TensorOpsTorch and TensorOpsTF."
            )

    def __call__(cls, *args, **kwargs) -> Union[TensorOpsTorch, TensorOpsTF]:
        framework = kwargs.get('framework', None)
        if framework is None and len(args) > 0:
            framework = args[0]

        if framework == "torch":
            specific_class = TensorOpsTorch
        elif framework == "tf":
            specific_class = TensorOpsTF
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        return specific_class(*args, **kwargs)


class TensorOps(Generic[T], metaclass=TensorOpsMeta):
    def __init__(self, framework: Literal["torch", "tf"]) -> None:
        self.framework: Literal["torch", "tf"] = framework
