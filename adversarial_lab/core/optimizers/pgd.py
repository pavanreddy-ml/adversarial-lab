from . import Optimizer


class PGD(Optimizer):
    def __init__(self,
                 learning_rate: float = 0.001,
                 projection_fn=None,
                 ) -> None:
        self.learning_rate = learning_rate
        self.projection_fn = projection_fn

    def initialize_optimizer(self):
        self.optimizer = self.tensor_ops.optimizers.PGD(
            learning_rate=self.learning_rate,
            projection_fn=self.projection_fn,
        )
