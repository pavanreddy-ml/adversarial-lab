import torch
import torch.nn.functional as F
from typing import List, Callable, Any, Union
import numpy as np

from adversarial_lab.core.types import TensorType, TensorVariableType, LossType, OptimizerType

class TensorOpsTorch:
    def __init__(self, *args, **kwargs) -> None:
        self.losses = TorchLosses()
        self.optimizers = TorchOptimizers()

    @staticmethod
    def tensor(arr: Union[np.ndarray, List[float], List[int], torch.Tensor]) -> torch.Tensor:
        """Convert numpy array or list to a PyTorch tensor."""
        return torch.tensor(arr, dtype=torch.float32)

    @staticmethod
    def variable(arr: Union[np.ndarray, List[float], List[int], torch.Tensor]) -> torch.Tensor:
        """Convert numpy array or list to a PyTorch tensor with gradients enabled."""
        return torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    @staticmethod
    def assign(tensor: torch.Tensor, value: Union[np.ndarray, List[float], List[int], torch.Tensor]) -> None:
        """Assign a new value to a PyTorch tensor (in-place update)."""
        with torch.no_grad():
            tensor.copy_(torch.tensor(value, dtype=torch.float32))

    @staticmethod
    def ones_like(tensor: torch.Tensor) -> torch.Tensor:
        """Create a tensor of ones with the same shape as the input tensor."""
        return torch.ones_like(tensor)

    @staticmethod
    def abs(tensor: torch.Tensor) -> torch.Tensor:
        """Return absolute values of elements in the tensor."""
        return torch.abs(tensor)

    @staticmethod
    def norm(tensor: torch.Tensor, p: float) -> torch.Tensor:
        """Compute the Lp norm of the tensor."""
        return torch.norm(tensor, p=p)

    @staticmethod
    def clip(tensor: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        """Clip tensor values between min and max."""
        return torch.clamp(tensor, min=min_val, max=max_val)

    @staticmethod
    def reduce_max(tensor: torch.Tensor, axis: Any | None = None, keepdims: bool = False) -> torch.Tensor:
        """Compute the maximum value in the tensor."""
        return torch.max(tensor, dim=axis, keepdim=keepdims)[0] if axis is not None else torch.max(tensor)

    @staticmethod
    def reduce_min(tensor: torch.Tensor, axis: Any | None = None, keepdims: bool = False) -> torch.Tensor:
        """Compute the minimum value in the tensor."""
        return torch.min(tensor, dim=axis, keepdim=keepdims)[0] if axis is not None else torch.min(tensor)

    @staticmethod
    def reduce_mean(tensor: torch.Tensor, axis: Any | None = None, keepdims: bool = False) -> torch.Tensor:
        """Compute the mean value in the tensor."""
        return torch.mean(tensor, dim=axis, keepdim=keepdims) if axis is not None else torch.mean(tensor)

    @staticmethod
    def reduce_sum(tensor: torch.Tensor, axis: Any | None = None, keepdims: bool = False) -> torch.Tensor:
        """Compute the sum of all elements in the tensor."""
        return torch.sum(tensor, dim=axis, keepdim=keepdims) if axis is not None else torch.sum(tensor)

    @staticmethod
    def random_normal(shape: List[int]) -> torch.Tensor:
        """Generate a tensor with random normal values."""
        return torch.randn(shape)

    @staticmethod
    def random_uniform(shape: List[int], minval: float, maxval: float) -> torch.Tensor:
        """Generate a tensor with random uniform values."""
        return torch.empty(shape).uniform_(minval, maxval)


class TorchLosses:
    def __init__(self):
        pass

    @staticmethod
    def binary_crossentropy(target: TensorType,
                            predictions: TensorType,
                            logits: TensorType,
                            from_logits: bool) -> TensorType:
        """Compute Binary Cross-Entropy (BCE) loss."""
        preds = logits if from_logits else predictions
        loss = F.binary_cross_entropy_with_logits(preds, target) if from_logits else F.binary_cross_entropy(preds, target)
        return loss.mean()

    @staticmethod
    def categorical_crossentropy(target: TensorType,
                                 predictions: TensorType,
                                 logits: TensorType,
                                 from_logits: bool) -> TensorType:
        """Compute Categorical Cross-Entropy (CCE) loss."""
        preds = logits if from_logits else predictions
        loss = F.cross_entropy(preds, target) if from_logits else -(target * preds.log()).sum(dim=1).mean()
        return loss

    @staticmethod
    def mean_absolute_error(target: TensorType,
                            predictions: TensorType) -> TensorType:
        """Compute Mean Absolute Error (MAE)."""
        loss = F.l1_loss(predictions, target, reduction='mean')
        return loss

    @staticmethod
    def mean_squared_error(target: TensorType,
                           predictions: TensorType) -> TensorType:
        """Compute Mean Squared Error (MSE)."""
        loss = F.mse_loss(predictions, target, reduction='mean')
        return loss
    

class TorchOptimizers:
    def __init__(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def Adam(learning_rate: float = 0.001,
             beta_1: float = 0.9,
             beta_2: float = 0.999,
             epsilon: float = 1e-8
             ) -> OptimizerType:
        """Apply Adam optimizer."""
        import torch.optim as optim
        temp_param = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
        opt = optim.Adam([temp_param], lr=learning_rate, betas=(beta_1, beta_2), eps=epsilon)
        opt.param_groups[0]["params"] = [] 
        return opt
    
    @staticmethod
    def SGD(learning_rate: float = 0.01,
            momentum: float = 0.0,
            nesterov: bool = False,
            weight_decay: float = 0.0,
            ) -> OptimizerType:
        """Apply SGD optimizer."""
        temp_param = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
        opt = torch.optim.SGD(
            [temp_param],
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay if weight_decay is not None else 0.0,
            nesterov=nesterov
        )
        opt.param_groups[0]["params"] = [] 
        return opt

    @staticmethod
    def PGD(learning_rate: float = 0.01,
            projection_fn: Callable[[TensorType], TensorType] = None
            ) -> OptimizerType:
        class PGDOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=0.01, projection_fn=None):
                defaults = dict(lr=lr, projection_fn=projection_fn)
                super().__init__(params, defaults)

            @torch.no_grad()
            def step(self, closure=None):
                for group in self.param_groups:
                    lr = group['lr']
                    projection_fn = group['projection_fn']
                    for param in group['params']:
                        if param.grad is None:
                            continue
                        param.data -= lr * param.grad
                        if projection_fn:
                            param.data = projection_fn(param.data)
                if closure is not None:
                    return closure()

        temp_param = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
        opt = PGDOptimizer([temp_param], lr=learning_rate, projection_fn=projection_fn)
        opt.param_groups[0]["params"] = [] 
        return opt

    @staticmethod
    def apply(optimizer: OptimizerType,
              variable_tensor: List[TensorType],
              gradients: List[TensorType]) -> None:
        """Apply gradients to update model weights."""
        if len(optimizer.param_groups[0]['params']) == 0:
            optimizer.add_param_group({'params': variable_tensor})

        for param, grad in zip(variable_tensor, gradients):
            param.grad = grad

        optimizer.step()
        optimizer.zero_grad()