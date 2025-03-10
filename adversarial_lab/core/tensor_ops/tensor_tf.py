import tensorflow as tf
import numpy as np

from typing import Union, List, Any
from adversarial_lab.core.types import TensorType, TensorVariableType, LossType, OptimizerType


class TensorOpsTF:
    def __init__(self, *args, **kwargs) -> None:
        self.losses = TFLosses()
        self.optimizers = TFOptimizers()

    @staticmethod
    def tensor(arr: Union[np.ndarray, List[float], List[int], TensorType]) -> TensorType:
        """Convert numpy array or list to a TensorFlow tensor."""
        return tf.convert_to_tensor(arr, dtype=tf.float32)

    @staticmethod
    def variable(arr: Union[np.ndarray, List[float], List[int], TensorType]) -> TensorVariableType:
        """Convert numpy array or list to a TensorFlow variable."""
        return tf.Variable(tf.convert_to_tensor(arr, dtype=tf.float32))

    @staticmethod
    def assign(tensor: TensorVariableType, value: Union[np.ndarray, List[float], List[int], TensorType]) -> None:
        """Assign a new value to a TensorFlow variable."""
        tensor.assign(tf.convert_to_tensor(value))

    @staticmethod
    def ones_like(tensor: TensorType) -> TensorType:
        """Create a tensor of ones with the same shape as the input tensor."""
        return tf.ones_like(tensor)

    @staticmethod
    def abs(tensor: TensorType) -> TensorType:
        """Return absolute values of elements in the tensor."""
        return tf.abs(tensor)

    @staticmethod
    def norm(tensor: TensorType, p: float) -> TensorType:
        """Compute the Lp norm of the tensor."""
        return tf.reduce_sum(tf.abs(tensor) ** p) ** (1.0 / p)

    @staticmethod
    def clip(tensor: TensorVariableType, min_val: float, max_val: float) -> TensorType:
        """Clip tensor values between min and max."""
        return tf.clip_by_value(tensor, min_val, max_val)
    
    @staticmethod
    def reduce_max(tensor: TensorType, axis: Any | None = None, keepdims: bool = False) -> TensorType:
        """Compute the maximum value in the tensor."""
        return tf.reduce_max(input_tensor=tensor, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def reduce_min(tensor: TensorType, axis: Any | None = None, keepdims: bool = False) -> TensorType:
        """Compute the minimum value in the tensor."""
        return tf.reduce_min(input_tensor=tensor, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def reduce_mean(tensor: TensorType, axis: Any | None = None, keepdims: bool = False) -> TensorType:
        """Compute the mean value in the tensor."""
        return tf.reduce_mean(input_tensor=tensor, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def reduce_sum(tensor: TensorType, axis: Any | None = None, keepdims: bool = False, name: Any | None = None) -> TensorType:
        """Compute the sum of all elements in the tensor."""
        return tf.reduce_sum(input_tensor=tensor, axis=axis, keepdims=keepdims, name=name)
    
    @staticmethod
    def random_normal(shape: List[int]) -> TensorType:
        """Generate a tensor with random normal values."""
        return tf.random.normal(shape)
    
    @staticmethod
    def random_uniform(shape: List[int], minval: float, maxval: float) -> TensorType:
        """Generate a tensor with random uniform values."""
        return tf.random.uniform(shape, minval=minval, maxval=maxval)
    


class TFLosses:
    def __init__(self):
        pass

    @staticmethod
    def binary_crossentropy(target: TensorType,
                            predictions: TensorType,
                            logits: TensorType,
                            from_logits: bool) -> TensorType:
        """Compute binary cross-entropy loss."""
        preds = logits if from_logits else predictions
        loss = tf.keras.losses.binary_crossentropy(
            target, preds, from_logits=from_logits)
        loss = tf.reduce_mean(loss)
        return loss

    @staticmethod
    def categorical_crossentropy(target: TensorType,
                                 predictions: TensorType,
                                 logits: TensorType,
                                 from_logits: bool) -> TensorType:
        """Compute Categorical cross-entropy loss."""
        preds = logits if from_logits else predictions
        loss = tf.keras.losses.categorical_crossentropy(
            target, preds, from_logits=from_logits)
        loss = tf.reduce_mean(loss)
        return loss

    @staticmethod
    def mean_absolute_error(target: TensorType,
                            predictions: TensorType
                            ) -> TensorType:
        """Compute Mean Absolute Error (MAE)."""
        loss_fn = tf.keras.losses.MeanAbsoluteError()
        loss = loss_fn(target, predictions)
        return tf.reduce_mean(loss)

    @staticmethod
    def mean_squared_error(target: TensorType,
                           predictions: TensorType
                           ) -> TensorType:
        """Compute Mean Absolute Error (MAE)."""
        loss_fn = tf.keras.losses.MeanSquaredError()
        loss = loss_fn(target, predictions)
        return tf.reduce_mean(loss)


class TFOptimizers:
    def __init__(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def Adam(learning_rate: float = 0.001,
             beta_1: float = 0.9,
             beta_2: float = 0.999,
             epsilon: float = 1e-8
             ) -> OptimizerType:
        """Create an Adam optimizer."""
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon
        )
    
    @staticmethod
    def SGD(learning_rate: float = 0.01,
            momentum: float = 0.0,
            nesterov: bool = False,
            weight_decay: float = None,
            ) -> OptimizerType:
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    
    @staticmethod
    def PGD(learning_rate: float = 0.001, projection_fn: Any = None) -> OptimizerType:
        class PGDOptimizer(tf.keras.optimizers.Optimizer):
            def __init__(self, learning_rate=0.01, projection_fn=None, name="PGD", **kwargs):
                if not isinstance(learning_rate, (float, int)):
                    raise ValueError(
                        f"Argument `learning_rate` must be a float or LearningRateSchedule, received: {type(learning_rate)}"
                    )

                super().__init__(name=name, **kwargs)
                self._learning_rate = self._build_learning_rate(learning_rate)
                self.projection_fn = projection_fn

            def _resource_apply_dense(self, grad, var, apply_state=None):
                var.assign_sub(self.lr * grad)
                if self.projection_fn:
                    var.assign(self.projection_fn(var))

            def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
                var.assign_sub(self.lr * grad)
                if self.projection_fn:
                    var.assign(self.projection_fn(var))

            def get_config(self):
                config = super().get_config()
                config.update({"learning_rate": float(self.lr.numpy())})
                return config

        return PGDOptimizer(learning_rate=learning_rate, projection_fn=projection_fn)
    
    @staticmethod
    def apply(optimizer: OptimizerType,
              variable_tensor: List[TensorVariableType],
              gradients: List[TensorType]) -> None:
        """Apply gradients to update model weights."""
        optimizer.apply_gradients(zip(gradients, variable_tensor))
        