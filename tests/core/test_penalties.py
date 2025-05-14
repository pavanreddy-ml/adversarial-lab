import pytest
import numpy as np
import tensorflow as tf
import torch

from adversarial_lab.core.penalties import *

@pytest.mark.parametrize("framework", ["torch", "tf"])
def test_penalty_instantiation(framework):
    class DummyPenalty(Penalty):
        def calculate(self, noise, *args, **kwargs):
            pass
    
    penalty = DummyPenalty()
    penalty.set_framework(framework)
    assert penalty.framework == framework


def test_penalty_invalid_framework():
    class DummyPenalty(Penalty):
        def calculate(self, noise, *args, **kwargs):
            pass
    
    with pytest.raises(ValueError, match="framework must be either 'tf', 'torch' or 'numpy'"):
        penalty = DummyPenalty()
        penalty.set_framework("invalid")


def test_penalty_has_calculate():
    class DummyPenalty(Penalty):
        def calculate(self, noise, *args, **kwargs):
            pass
    
    penalty = DummyPenalty()
    assert hasattr(penalty, "calculate")


@pytest.mark.parametrize("framework", ["torch", "tf"])
@pytest.mark.parametrize("p", [1, 2, 3])
def test_lp_norm(framework, p):
    penalty_fn = LpNorm(p=p, lambda_val=1.0)
    penalty_fn.set_framework(framework)

    if framework == "tf":
        noise = tf.constant([1.0, -2.0, 3.0], dtype=tf.float32)
    else:
        noise = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)

    try:
        penalty = penalty_fn.calculate(noise)
    except NotImplementedError:
        pass
    assert penalty_fn.value is not None


@pytest.mark.parametrize("function_type", ["valid", "missing_noise"])
def test_penalty_from_function(function_type):
    if function_type == "missing_noise":
        def invalid_function(*args, **kwargs):
            return args

        with pytest.raises(TypeError, match="must have parameter: 'noise'"):
            PenaltyFromFunction.create(invalid_function)

    if function_type == "valid":
        def valid_function(noise, *args, **kwargs):
            return noise * 2

        penalty = PenaltyFromFunction.create(valid_function)

        noise = tf.constant([1.0, -1.0, 0.5], dtype=tf.float32)
        result = penalty.calculate(noise)

        assert np.array_equal(result.numpy(), [2.0, -2.0, 1.0])
        