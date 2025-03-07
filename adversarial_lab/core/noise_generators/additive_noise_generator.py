from typing import Literal, List, Union

import numpy as np

from .noise_generator_base import NoiseGenerator

from adversarial_lab.core.types import TensorType, TensorVariableType


class AdditiveNoiseGenerator(NoiseGenerator):
    def __init__(self,
                 scale: List[int] = (-1, 1),
                 dist: Literal["normal", "uniform"] = "normal",
                 mask: TensorType | np.ndarray = None,
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(mask=mask)

        self.dist = dist
        self.scale = scale

    def generate_noise_meta(self,
                            sample: Union[np.ndarray, TensorType],
                            ) -> TensorVariableType:
        super().generate_noise_meta(sample)

        shape = sample.shape
        if shape[0] is None or shape[0] == 1:
            shape = shape[1:]
            
        if self.dist == "normal":
            noise_meta = self.tensor_ops.random_normal(shape)
            noise_meta = self.scale[0] + (self.scale[1] - self.scale[0]) * (noise_meta - self.tensor_ops.reduce_min(
                noise_meta)) / (self.tensor_ops.reduce_max(noise_meta) - self.tensor_ops.reduce_min(noise_meta))
        elif self.dist == "uniform":
            noise_meta = self.tensor_ops.random_uniform(
                shape, minval=self.scale[0], maxval=self.scale[1])
        else:
            raise ValueError(f"Unsupported distribution: {self.dist}")
        
        self._mask = self.tensor_ops.ones_like(noise_meta) if self._mask is None else self.tensor_ops.tensor(self._mask)

        noise_meta = self.tensor_ops.variable(noise_meta)

        noise_meta.assign(noise_meta * self._mask)
        return [noise_meta]

    def get_noise(self,
                  noise_meta: List[TensorVariableType]
                  ) -> np.ndarray:
        return noise_meta[0].numpy()

    def apply_noise(self,
                    sample: TensorType | np.ndarray,
                    noise_meta: List[TensorVariableType]
                    ) -> TensorType:
        return sample + (self._mask * noise_meta[0])
