from . import BackpropagationBase, BackpropagationTensorFlow, BackpropagationTorch
from adversary.losses import Loss, LossRegistry

import torch
import tensorflow as tf

class Backpropagation(BackpropagationBase):
    def __init__(self, 
                 model,
                 loss) -> None:
        if isinstance(model, torch.nn.Module):
            self.framework = "torch"
        elif isinstance(model, tf.keras.Model):
            self.framework = "tensorflow"
        else:
            raise ValueError("Unsupported model type. Must be a PyTorch or TensorFlow model.")

        if self.framework == "torch":
            self.backpropagation_class = BackpropagationTorch()
        elif self.framework == "tf":
            self.backpropagation_class = BackpropagationTensorFlow()

        if isinstance(loss, str):
            self.loss = LossRegistry.get(loss)
        elif isinstance(loss, Loss):
            self.loss = loss
        else:
            raise TypeError(f"Invalid type for loss: '{type(loss)}'")

    def run(self):
        self.backpropagation_class.run()