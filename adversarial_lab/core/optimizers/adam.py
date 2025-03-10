from . import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer class for optimizing model parameters.
    """
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8
                 ) -> None:
        """
        Initialize the Adam optimizer with specified hyperparameters.

        Args:
            learning_rate (float): The step size used for updating weights. Default is 0.001.
            beta1 (float): The exponential decay rate for the first moment estimates. Default is 0.9.
            beta2 (float): The exponential decay rate for the second moment estimates. Default is 0.999.
            epsilon (float): A small constant added for numerical stability. Default is 1e-8.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def initialize_optimizer(self):
        self.optimizer = self.tensor_ops.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta1,
            beta_2=self.beta2,
            epsilon=self.epsilon
        )
