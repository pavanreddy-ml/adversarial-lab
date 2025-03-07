from . import Optimizer


class SGD(Optimizer):
    def __init__(self,
                 learning_rate: float = 0.01,
                    momentum: float = 0.0,
                    nesterov: bool = False,
                    weight_decay: float = None,
                 ) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay

    def initialize_optimizer(self):
        self.optimizer = self.tensor_ops.optimizers.SGD(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov,
            weight_decay=self.weight_decay
        )
