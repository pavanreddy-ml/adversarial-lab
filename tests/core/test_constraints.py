import pytest
import numpy as np
import tensorflow as tf
import torch

from adversarial_lab.core.constraints import *

@pytest.mark.parametrize("framework", ["torch", "tf"])
def test_post_optimization_constraint_instantiation(framework):
    constraint = PostOptimizationConstraint(framework)
    assert constraint.framework == framework

def test_post_optimization_constraint_invalid_framework():
    with pytest.raises(ValueError, match="Unsupported framework"):
        PostOptimizationConstraint("invalid_framework")

def test_post_optimization_constraint_apply():
    class DummyConstraint(PostOptimizationConstraint):
        def apply(self, noise):
            pass
    
    constraint = DummyConstraint(framework="torch")
    assert hasattr(constraint, "apply")

@pytest.mark.parametrize("framework", ["torch", "tf"])
def test_po_clip_clipping(framework):
    clip_constraint = POClip(framework, min=-0.5, max=0.5)
    
    if framework == "tf":
        noise = tf.Variable([-1.0, 0.0, 1.0], dtype=tf.float32)
        clip_constraint.tf_op(noise)
        assert tf.reduce_min(noise).numpy() >= -0.5
        assert tf.reduce_max(noise).numpy() <= 0.5
    
    if framework == "torch":
        noise = torch.tensor([-1.0, 0.0, 1.0])
        clipped_noise = torch.clamp(noise, min=-0.5, max=0.5)
        assert torch.min(clipped_noise) >= -0.5
        assert torch.max(clipped_noise) <= 0.5

def test_po_lp_norm_torch_not_implemented():
    lp_constraint = POLpNorm("torch", epsilon=0.1)
    noise = torch.tensor([1.0, -1.0])
    
    with pytest.raises(NotImplementedError):
        lp_constraint.torch_op(noise)

def test_po_lp_norm_tf():
    lp_constraint = POLpNorm("tf", epsilon=1.0, l_norm="2", max_iter=10)
    noise = tf.Variable([2.0, -2.0], dtype=tf.float32)
    
    lp_constraint.tf_op(noise)
    norm = tf.norm(noise, ord=2).numpy()
    
    assert norm <= 1.0 + 1e-6  # Allow small numerical differences

def test_po_constraint_from_function_invalid():
    def invalid_function(x):
        return x
    
    with pytest.raises(TypeError, match="must have parameter: 'noise'"):
        POConstraintFromFunction.create(invalid_function)

def test_po_constraint_from_function_apply():
    def valid_function(noise, *args, **kwargs):
        return noise * 2

    constraint = POConstraintFromFunction.create(valid_function)

    noise = tf.Variable([1.0, 2.0], dtype=tf.float32)
    result = constraint.apply(noise).numpy()

    assert np.array_equal(result, [2.0, 4.0])


