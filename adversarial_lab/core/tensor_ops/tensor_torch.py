import torch
import torch.nn.functional as F
from typing import List, Callable

from adversarial_lab.core.types import TensorType, TensorVariableType, LossType, OptimizerType

class TensorOpsTorch:
    def __init__(self, *args, **kwargs) -> None:
        self.losses = TorchLosses()


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
    def adam(learning_rate: float = 0.001,
             beta1: float = 0.9,
             beta2: float = 0.999,
             epsilon: float = 1e-8
             ) -> OptimizerType:
        """Apply Adam optimizer."""
        return torch.optim.Adam([], lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
    
    @staticmethod
    def sgd(learning_rate: float = 0.01,
            momentum: float = 0.0,
            weight_decay: float = 0.0,
            nesterov: bool = False
            ) -> OptimizerType:
        """Apply SGD optimizer."""
        return torch.optim.SGD(
            [],
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )

    @staticmethod
    def pgd(learning_rate: float = 0.01,
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

        return PGDOptimizer([], lr=learning_rate, projection_fn=projection_fn)

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