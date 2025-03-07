import pytest
import numpy as np
import tensorflow as tf
import torch

from adversarial_lab.core.tensor_ops.tensor_tf import TensorOpsTF
from adversarial_lab.core.tensor_ops.tensor_torch import TensorOpsTorch


def test_tensor_ops_method_consistency():
    """Test that TensorOpsTorch and TensorOpsTF have the same method signatures."""
    torch_methods = set(dir(TensorOpsTorch))
    tf_methods = set(dir(TensorOpsTF))

    # Remove dunder methods
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
