from typing import Literal, List, Tuple, Optional

from . import Masking

import warnings
import numpy as np


class PositionalMasking(Masking):
    def __init__(self,
                 limits: List[Optional[Tuple[Optional[float], Optional[float]]]],
                 strategy: Literal["absolute", "relative"] = "relative",
                 ):
        self.strategy = strategy
        self._validate_and_set_limits(limits)

    def create(self, sample):
        self._validate_shape(sample)
        if self.strategy == "relative":
            return self._create_relative_mask(sample)
        elif self.strategy == "absolute":
            return self._create_absolute_mask(sample)
        
    def _create_absolute_mask(self, sample):
        s = self._get_unbatched_sample(sample)
        mask = np.zeros(s.shape, dtype=np.uint8)
        slices = []
        for i, limit in enumerate(self.limits):
            s_min, s_max = 0, s.shape[i]

            if limit is None:
                l_min, l_max = s_min, s_max
            else:
                l_min, l_max = limit

            if l_min < s_min:
                warnings.warn(f"l_min ({l_min}) is less than the minimum index ({s_min}). Setting l_min to {s_min}.")
                l_min = s_min
            if l_max > s_max:
                warnings.warn(f"l_max ({l_max}) is greater than the maximum index ({s_max}). Setting l_max to {s_max}.")
                l_max = s_max

            slices.append(slice(int(l_min), int(l_max)))

        mask[tuple(slices)] = 1
        return self.tensor_ops.tensor(mask)

    def _create_relative_mask(self, sample):
        s = self._get_unbatched_sample(sample)
        mask = np.zeros(s.shape, dtype=np.uint8)
        slices = []

        for i, limit in enumerate(self.limits):
            if limit is None:
                l_min, l_max = 0, 1
            else:
                l_min, l_max = s.shape[i] * limit[0], s.shape[i] * limit[1]

            slices.append(slice(int(l_min), int(l_max)))

        mask[tuple(slices)] = 1
        return self.tensor_ops.tensor(mask)

    def _validate_and_set_limits(self, limits):
        if any(l is not None and not isinstance(l, (list, tuple)) for l in limits):
            raise ValueError("Each limit must be None or a list/tuple with exactly two elements l_min and l_max.")

        if any(l is not None and len(l) != 2 for l in limits):
            raise ValueError("Each limit must contain exactly two elements l_min and l_max, or be None.")
        
        if any(l is not None and (l[0] is None or l[1] is None) for l in limits):
            raise ValueError("Both l_min and l_max must be provided. None values are not allowed.")

        if any(l is not None and l[0] is not None and l[1] is not None and l[0] >= l[1] for l in limits):
            raise ValueError("l_min must be less than l_max for each limit.")

        if self.strategy == "relative":
            if any(l is not None and l[0] is not None and l[1] is not None and (0 <= l[0] <= 1 or 0 <= l[1] <= 1) for l in limits):
                raise ValueError("For relative strategy, l_min and l_max must be outside the range [0, 1].")

        if self.strategy == "absolute":
            if any(l is not None and l[0] is not None and l[1] is not None and (not isinstance(l[0], int) or not isinstance(l[1], int)) for l in limits):
                raise ValueError("For absolute strategy, l_min and l_max must be integers.")
            
        self.limits = limits
            
    def _validate_shape(self, sample):
        s = self.tensor_ops.remove_batch_dim(sample)
        if len(s.shape) != len(self.limits):
            raise ValueError(f"Limits for masking must match the number of dimensions in the sample. Expected {self.limits}, got {len(sample.shape)}.")

    def _get_unbatched_sample(self, sample: np.ndarray) -> np.ndarray:
        if sample.shape[0] == 1 or sample.shape[0] == None:
            return sample[0]
        return sample