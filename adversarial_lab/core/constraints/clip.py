from . import PostOptimizationConstraint
from adversarial_lab.core.types import TensorVariableType


class POClip(PostOptimizationConstraint):
    """
    Post-optimization constraint that clips the adversarial noise within a specified range.

    This constraint ensures that all elements of the noise tensor remain within the 
    specified `[min, max]` range.
    """

    def __init__(self, min: float = -1.0, max: float = 1.0) -> None:
        """
        Initialize the POClip constraint.

        Parameters:
            min (float): The minimum value to clip the noise to.
            max (float): The maximum value to clip the noise to.

        Raises:
            ValueError: If `min` is greater than `max`.

        Notes:
            - If `min` is greater than `max`, an error is raised.
            - The noise tensor is clipped element-wise using `tensor_ops.clip()`.
        """
        if min > max:
            raise ValueError("min cannot be greater than max.")

        super().__init__()
        self.min = min
        self.max = max

    def apply(self, 
              noise: TensorVariableType, 
              *args,
              **kwargs
              ) -> None:
        clipped_value = self.tensor_ops.clip(noise, self.min, self.max)
        self.tensor_ops.assign(noise, clipped_value)

