from . import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer class for optimizing model parameters.
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 nesterov: bool = False,
                 weight_decay: float = None,
                 ) -> None:
        """
        Initialize the SGD optimizer with the specified hyperparameters.

        Args:
            learning_rate (float): The step size used for updating weights. Default is 0.01.
            momentum (float): A factor that accelerates SGD in relevant directions and dampens oscillations. Default is 0.0.
            nesterov (bool): Whether to use Nesterov accelerated gradient. Default is False.
            weight_decay (float, optional): A factor for L2 regularization, which prevents overfitting. Default is None.
        """
        super().__init__()
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
