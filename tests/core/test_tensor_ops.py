import pytest
import numpy as np
import tensorflow as tf
import torch

from adversarial_lab.core.tensor_ops.tensor_tf import TensorOpsTF
from adversarial_lab.core.tensor_ops.tensor_torch import TensorOpsTorch
from adversarial_lab.core.tensor_ops.tensor_ops import TensorOps



def test_tensor_ops_method_consistency():
    """Test that TensorOpsTorch and TensorOpsTF have the same method signatures."""
    torch_methods = set(dir(TensorOpsTorch))
    tf_methods = set(dir(TensorOpsTF))

    torch_methods = {m for m in torch_methods if not m.startswith("__")}
    tf_methods = {m for m in tf_methods if not m.startswith("__")}

    torch_only = torch_methods - tf_methods
    tf_only = tf_methods - torch_methods

    error_messages = []

    if torch_only:
        error_messages.append(f"Methods in TensorOpsTorch not in TensorOpsTF: {sorted(torch_only)}")
    if tf_only:
        error_messages.append(f"Methods in TensorOpsTF not in TensorOpsTorch: {sorted(tf_only)}")

    assert not torch_only and not tf_only, "\n".join(error_messages)

TEST_VALUES = [
    ([1, 2, 3],),
    ([[-1, 0, 1], [2, -2, 3]],),
    ([[-0.5, 0.5], [1.5, -1.5]],),
    ([1, 2, 3, 4, 5],),
    ([[-1, 0], [1, 2], [-3, 4]],),
]

@pytest.mark.parametrize("input_data", TEST_VALUES)
@pytest.mark.parametrize("framework", ["tf", "torch"])
def test_tensor_ops_tf(input_data, framework):
    tensor_ops = TensorOps(framework)

    np_arr = np.array(input_data, dtype=np.float32)
    tensor = tensor_ops.tensor(np_arr)

    assert np.allclose(tensor_ops.abs(tensor).numpy(), np.abs(np_arr))
    assert np.allclose(tensor_ops.norm(tensor, 2).numpy(), np.linalg.norm(np_arr))
    assert np.allclose(tensor_ops.ones_like(tensor).numpy(), np.ones_like(np_arr))
    assert np.allclose(tensor_ops.reduce_max(tensor).numpy(), np.max(np_arr))
    assert np.allclose(tensor_ops.reduce_min(tensor).numpy(), np.min(np_arr))
    assert np.allclose(tensor_ops.reduce_mean(tensor).numpy(), np.mean(np_arr))
    assert np.allclose(tensor_ops.reduce_sum(tensor).numpy(), np.sum(np_arr))

    for axis in [None, 0, 1]:
        if axis is not None and len(np_arr.shape) <= axis:
            continue
        assert np.allclose(tensor_ops.reduce_max(tensor, axis=axis).numpy(), np.max(np_arr, axis=axis))
        assert np.allclose(tensor_ops.reduce_min(tensor, axis=axis).numpy(), np.min(np_arr, axis=axis))
        assert np.allclose(tensor_ops.reduce_mean(tensor, axis=axis).numpy(), np.mean(np_arr, axis=axis))
        assert np.allclose(tensor_ops.reduce_sum(tensor, axis=axis).numpy(), np.sum(np_arr, axis=axis))

    clipped = tensor_ops.clip(tensor, -0.5, 0.5).numpy()
    assert np.allclose(clipped, np.clip(np_arr, -0.5, 0.5))
