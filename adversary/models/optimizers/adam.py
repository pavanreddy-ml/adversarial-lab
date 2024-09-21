from typing import Literal, Union, List
import torch
import tensorflow as tf
from torch.optim import Adam as TorchAdam
from tensorflow.keras.optimizers import Adam as TFAdam
from . import Optimizer

class Adam(Optimizer):
    def __init__(self, 
                 framework: Literal["torch", "tf"],
                 learning_rate: float = 0.001,
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

    def torch_apply(self, 
                    model_weights: List[torch.Tensor], 
                    gradients: List[torch.Tensor]) -> None:
        for param, grad in zip(model_weights, gradients):
            param.grad = grad

        self.optimizer.step()
        self.optimizer.zero_grad()

    def tf_apply(self, 
                 model_weights: List[tf.Variable], 
                 gradients: List[tf.Tensor]) -> None:
        self.optimizer.apply_gradients(zip(gradients, model_weights))

