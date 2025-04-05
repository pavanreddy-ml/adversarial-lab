from . import Masking

import numpy as np

class CustomMasking(Masking):
    def __init__(self, mask: np.ndarray):
        self.mask = mask

    def create(self, sample):
        s = self.tensor_ops.remove_batch_dim(sample)
        if s.shape != self.mask.shape:
            raise ValueError(f"The shape of the mask must match the shape of the sample. Expected shape: {s.shape}, got: {self.mask.shape}")
        return self.tensor_ops.tensor(self.mask)
    