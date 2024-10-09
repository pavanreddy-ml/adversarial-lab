from typing import Literal, Union, List
import torch
import tensorflow as tf
from torch.optim import Adam as TorchAdam
from tensorflow.keras.optimizers import Adam as TFAdam
from . import Optimizer

class Adam(Optimizer):
    def __init__(self, 
                 framework: Literal["torch", "tf"],
                 learning_rate: float = 0.01,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8
                 ) -> None:
        self.framework = framework
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        if self.framework == "torch":
            self.optimizer = TorchAdam([], lr=self.learning_rate, betas=(self.beta1, self.beta2), eps=self.epsilon)
        elif self.framework == "tf":
            self.optimizer = TFAdam(learning_rate=self.learning_rate, beta_1=self.beta1, beta_2=self.beta2, epsilon=self.epsilon)
        else:
            raise ValueError("Framework must be either 'torch' or 'tf'")
        
    def apply(self,
              model_weights: torch.Tensor,
              gradients: torch.Tensor):
        super().apply(model_weights, gradients)

    def torch_op(self, 
                    model_weights: List[torch.Tensor], 
                    gradients: List[torch.Tensor]) -> None:
        if len(self.optimizer.param_groups[0]['params']) == 0:
            self.optimizer.add_param_group({'params': model_weights})

        for param, grad in zip(model_weights, gradients):
            param.grad = grad

        self.optimizer.step()
        self.optimizer.zero_grad()

    def tf_op(self, 
                 model_weights: List[tf.Variable], 
                 gradients: List[tf.Tensor]) -> None:
        self.optimizer.apply_gradients(zip(gradients, model_weights))

