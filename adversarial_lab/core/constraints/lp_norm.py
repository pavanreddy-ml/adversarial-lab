from . import PostOptimizationConstraint
from adversarial_lab.core.types import TensorVariableType


class POLpNorm(PostOptimizationConstraint):
    """
    Post-optimization constraint that enforces an Lp-norm bound on adversarial noise.

    This constraint ensures that the perturbation applied to the input adheres to a given 
    Lp-norm constraint by performing a binary search to scale the noise appropriately.
    """

    def __init__(self, 
                 epsilon: float = -1.0,
                 l_norm: str = "2",
                 max_iter: int = 100) -> None:
        """
        Initialize the POLpNorm constraint.

        Parameters:
            epsilon (float): The maximum allowed Lp-norm of the noise. 
                If `-1.0`, no constraint is enforced.
            l_norm (str): The Lp-norm type to enforce, must be one of {"1", "2" ... "p"}.
            max_iter (int): The maximum number of iterations to perform during binary search.

        Raises:
            ValueError: If `l_norm` is not one of {"1", "2", "inf"}.
        """
        self.epsilon = epsilon
        self.l_norm = l_norm
        self.max_iter = max_iter
    
    def apply(self, 
              noise: TensorVariableType, 
              *args,
              **kwargs
              ) -> None:
        def compute_lp_norm(tensor, p):
            return self.tensor_ops.reduce_sum(self.tensor_ops.abs(tensor) ** p) ** (1.0 / p)

        p = float(self.l_norm)
        low, high = 0.0, 1.0

        for _ in range(self.max_iter):
            scale_factor = (low + high) / 2.0
            scaled_noise = noise * scale_factor
            norm = compute_lp_norm(scaled_noise, p)

            if norm > self.epsilon:
                high = scale_factor
            else:
                low = scale_factor

            if self.tensor_ops.abs(norm - self.epsilon) < 1e-6:
                break

        self.tensor_ops.assign(noise, noise * low)
