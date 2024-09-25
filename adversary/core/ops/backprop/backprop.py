from . import BackpropagationBase, BackpropagationTensorFlow, BackpropagationTorch


import torch
import tensorflow as tf

class Backpropagation(BackpropagationBase):
    def __init__(self, 
                 framework,
                 loss) -> None:
        self.framework = framework
        self.loss = loss

        if self.framework == "torch":
            self.backpropagation_class = BackpropagationTorch()
        elif self.framework == "tf":
            self.backpropagation_class = BackpropagationTensorFlow()

    def run(self):
        self.backpropagation_class.run()