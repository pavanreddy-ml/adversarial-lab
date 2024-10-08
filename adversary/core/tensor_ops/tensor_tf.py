import tensorflow as tf
from typing import Optional, Dict, Any, Union

class TensorOpsTF:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def tensor(self, 
               shape: Union[tuple, list, tf.TensorShape], 
               distribution: str = "fill", 
               dist_params: Optional[Dict[str, Union[int, float]]] = None) -> tf.Tensor:

        if distribution == "fill":
            if dist_params is None:
                fill_value = 0
            else:
                fill_value = dist_params['value']
            return tf.fill(shape, fill_value)

        elif distribution == "uniform":
            if dist_params is None:
                min_val = 0.0
                max_val = 1.0
            else:
                if 'min' not in dist_params or 'max' not in dist_params:
                    raise ValueError("For 'uniform' distribution, 'min' and 'max' must be specified in dist_params")
                min_val = dist_params['min']
                max_val = dist_params['max']
                if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)) or min_val >= max_val:
                    raise ValueError("For 'uniform' distribution, 'min' must be less than 'max' and both must be numeric")
            return tf.random.uniform(shape, minval=min_val, maxval=max_val)

        elif distribution == "normal":
            if dist_params is None:
                mean = 0.0
                stddev = 1.0
            else:
                if 'mean' not in dist_params or 'stddev' not in dist_params:
                    raise ValueError("For 'normal' distribution, 'mean' and 'stddev' must be specified in dist_params")
                mean = dist_params['mean']
                stddev = dist_params['stddev']
                if not isinstance(mean, (int, float)) or not isinstance(stddev, (int, float)) or stddev <= 0:
                    raise ValueError("For 'normal' distribution, 'stddev' must be greater than 0 and both must be numeric")
            return tf.random.normal(shape, mean=mean, stddev=stddev)

        elif distribution == "linspace":
            if dist_params is None:
                start = 0.0
                end = 1.0
                num = shape[0] if len(shape) > 0 else 50
            else:
                if 'start' not in dist_params or 'end' not in dist_params or 'num' not in dist_params:
                    raise ValueError("For 'linspace' distribution, 'start', 'end', and 'num' must be specified in dist_params")
                start = dist_params['start']
                end = dist_params['end']
                num = dist_params['num']
                if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not isinstance(num, int) or num <= 0:
                    raise ValueError("For 'linspace' distribution, 'start' and 'end' must be numeric, and 'num' must be a positive integer")
            return tf.linspace(start, end, num)

        else:
            raise ValueError(f"Unsupported random distribution: {distribution}")

    def tensor_like(self, 
                    array: Union[tf.Tensor, list, tuple], 
                    distribution: str = "fill", 
                    dist_params: Optional[Dict[str, Union[int, float]]] = None) -> tf.Tensor:
        shape = tf.shape(array)
        return self.tensor(shape, distribution, dist_params)

    def to_tensor(self, 
                  data: Any, 
                  dtype: tf.DType = tf.float32) -> tf.Tensor:
        return tf.convert_to_tensor(data, dtype=dtype)
