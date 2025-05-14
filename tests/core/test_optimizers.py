import pytest
import tensorflow as tf
import torch

from adversarial_lab.core.optimizers import *
from adversarial_lab.core.optimizers import OptimizerRegistry


@pytest.mark.parametrize("framework", ["torch", "tf"])
def test_optimizer_instantiation(framework):
    class DummyOptimizer(Optimizer):
        def initialize_optimizer(self):
            self.optimizer = None
        
        def update(self, weights, gradients):
            pass
    
    optimizer = DummyOptimizer()
    optimizer.set_framework(framework)
    assert optimizer.framework == framework


def test_optimizer_invalid_framework():
    class DummyOptimizer(Optimizer):
        def initialize_optimizer(self):
            self.optimizer = None
        
        def update(self, weights, gradients):
            pass
    
    with pytest.raises(ValueError, match="framework must be either 'tf', 'torch' or 'numpy'"):
        optimizer = DummyOptimizer()
        optimizer.set_framework("invalid_framework")


@pytest.mark.parametrize("framework", ["torch", "tf"])
@pytest.mark.parametrize("optimizer_class", [Adam, SGD, PGD])
def test_optimizer_initialization(framework, optimizer_class):
    optimizer = optimizer_class()
    optimizer.set_framework(framework)
    
    assert optimizer.framework == framework
    assert optimizer.optimizer is not None


@pytest.mark.parametrize("framework", ["torch", "tf"])
@pytest.mark.parametrize("optimizer_class", [Adam, SGD, PGD])
def test_optimizer_update(framework, optimizer_class):
    optimizer = optimizer_class()
    optimizer.set_framework(framework)

    if framework == "tf":
        weights = [tf.Variable([1.0, 2.0, 3.0], dtype=tf.float32)]
        gradients = [tf.constant([0.1, 0.2, 0.3], dtype=tf.float32)]
    else:
        weights = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
        gradients = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    try:
        optimizer.update(weights, gradients)
    except NotImplementedError:
        pass

    assert optimizer.optimizer is not None


@pytest.mark.parametrize("optimizer_name", ["adam", "sgd", "pgd"])
def test_optimizer_registry(optimizer_name):
    optimizer_class = OptimizerRegistry.get(optimizer_name)
    assert issubclass(optimizer_class, Optimizer)
