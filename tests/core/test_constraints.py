import pytest
import numpy as np
import tensorflow as tf
import torch

from adversarial_lab.core.constraints import *

@pytest.mark.parametrize("framework", ["torch", "tf"])
def test_post_optimization_constraint_instantiation(framework):
    class DummyConstraint(PostOptimizationConstraint):
        def apply(self, noise):
            pass
    
    constraint = DummyConstraint()

    constraint.set_framework(framework)
    assert constraint.framework == framework

def test_post_optimization_constraint_invalid_framework():
    class DummyConstraint(PostOptimizationConstraint):
            def apply(self, noise):
                pass

    with pytest.raises(ValueError, match="framework must be either 'tf' or 'torch'"):
        constraint = DummyConstraint()
        constraint.set_framework("invalid")

def test_post_optimization_constraint_apply():
    class DummyConstraint(PostOptimizationConstraint):
        def apply(self, noise):
            pass
    
    constraint = DummyConstraint()
    assert hasattr(constraint, "apply")

@pytest.mark.parametrize("framework", ["torch", "tf"])
def test_po_clip_clipping(framework):
    if framework == "tf":
        clip_constraint = POClip(min=-0.5, max=0.5)
        clip_constraint.set_framework("tf")
        noise = tf.Variable([-1.0, 0.0, 1.0], dtype=tf.float32)
        try:
            clip_constraint.apply(noise)
        except NotImplementedError:
            pass
        assert tf.reduce_min(noise).numpy() >= -0.5
        assert tf.reduce_max(noise).numpy() <= 0.5
    
    if framework == "torch":
        clip_constraint = POClip(min=-0.5, max=0.5)
        clip_constraint.set_framework("torch")
        noise = torch.tensor([-1.0, 0.0, 1.0])
        try:
            clip_constraint.apply(noise)
        except NotImplementedError:
            pass
        assert torch.min(noise) >= -0.5
        assert torch.max(noise) <= 0.5

@pytest.mark.parametrize("framework", ["torch", "tf"])
def test_po_lp_norm_apply(framework):   
    if framework == "tf":
        lp_constraint = POLpNorm(epsilon=1.0, l_norm="2", max_iter=10)
        lp_constraint.set_framework("tf")
        noise = tf.Variable([2.0, -2.0], dtype=tf.float32)
        try:
            lp_constraint.apply(noise)
        except NotImplementedError:
            pass
        norm = tf.norm(noise, ord=2).numpy()
        assert norm <= 1.0 + 1e-6

    if framework == "torch":
        lp_constraint = POLpNorm(epsilon=1.0, l_norm="2", max_iter=10)
        lp_constraint.set_framework("torch")
        noise = torch.tensor([2.0, -2.0], dtype=torch.float32)
        try:
            lp_constraint.apply(noise)
        except NotImplementedError:
            pass
        norm = torch.norm(noise, p=2).item()
        assert norm <= 1.0 + 1e-6

@pytest.mark.parametrize("function_type", ["valid", "invalid"])
def test_po_constraint_from_function(function_type):
    if function_type == "invalid":
        def invalid_function(x):
            return x
        
        with pytest.raises(TypeError, match="must have parameter: 'noise'"):
            POConstraintFromFunction.create(invalid_function)

    if function_type == "valid":
        def valid_function(noise, *args, **kwargs):
            return noise * 2

        constraint = POConstraintFromFunction.create(valid_function)
        noise = tf.Variable([1.0, 2.0], dtype=tf.float32)
        result = constraint.apply(noise).numpy()

        assert np.array_equal(result, [2.0, 4.0])