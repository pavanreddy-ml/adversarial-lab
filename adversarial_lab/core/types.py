from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    try:
        import tensorflow as tf
        from adversarial_lab.core.tensor_ops.tensor_tf import TensorOpsTF
    except ImportError:
        tf = None

    try:
        import torch
        from adversarial_lab.core.tensor_ops.tensor_torch import TensorOpsTorch
    except ImportError:
        torch = None

TensorType = Union["tf.Tensor", "torch.Tensor"]
TensorVariableType = Union["tf.Variable", "torch.Tensor"]
LossType = Union["tf.keras.losses.Loss", "torch.nn.modules.loss._Loss", TensorType]
OptimizerType = Union["tf.optimizers.Optimizer", "torch.optim.Optimizer"]
ModelType = Union["tf.keras.Model", "torch.nn.Module", "tf.keras.Sequential"]
TensorOpsType = Union["TensorOpsTorch", "TensorOpsTF"]

