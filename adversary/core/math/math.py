from abc import ABCMeta
from typing import Literal
import torch
import tensorflow as tf

from .math_tf import MathTF
from .math_torch import MathTorch


class MathMeta(ABCMeta):
    def __new__(cls, name, bases, dct):
        cls.validate_methods_consistency()
        return super().__new__(cls, name, bases, dct)

    @staticmethod
    def validate_methods_consistency():
        torch_methods = set(dir(MathTorch))
        tf_methods = set(dir(MathTF))

        torch_methods = {m for m in torch_methods if not m.startswith("__")}
        tf_methods = {m for m in tf_methods if not m.startswith("__")}

        torch_only = torch_methods - tf_methods
        tf_only = tf_methods - torch_methods

        if torch_only or tf_only:
            error_message = "Method inconsistency found between MathTorch and MathTF.\n"

            if torch_only:
                error_message += f"Methods in MathTorch not in MathTF: {torch_only}\n"
            if tf_only:
                error_message += f"Methods in MathTF not in MathTorch: {tf_only}\n"

            raise NotImplementedError(
                error_message + 
                "For consistency, please implement these methods in both MathTorch and MathTF."
            )

    def __call__(cls, *args, **kwargs):
        framework = kwargs.get('framework', None)
        if framework is None and len(args) > 0:
            framework = args[0]

        if framework == "torch":
            cls = type(cls.__name__, (MathTorch, cls), {})
        elif framework == "tf":
            cls = type(cls.__name__, (MathTF, cls), {})
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        return super(MathMeta, cls).__call__(*args, **kwargs)


class Math(metaclass=MathMeta):
    def __init__(self,
                 framework: Literal["torch", "tf"]
                 ) -> None:
        self.framework = framework


