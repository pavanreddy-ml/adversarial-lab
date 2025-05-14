import numpy as np
from copy import deepcopy

from . import GradientEstimator
from adversarial_lab.core.losses import Loss

from typing import Callable, List


class FiniteDifferenceGE(GradientEstimator):
    def __init__(self,
                 epsilon: float = 1e-5,
                 batch_size: int = 32,
                 block_size: int = 1):
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.block_size = block_size

    def calculate(self,
                  sample: np.ndarray,
                  noise: List[np.ndarray],
                  target_vector: np.ndarray,
                  predict_fn: Callable,
                  construct_perturbation_fn: Callable,
                  loss: Loss,
                  *args,
                  **kwargs) -> List[np.ndarray]:

        grads_wrt_noise = []

        for i, n in enumerate(noise):
            grad_i = np.zeros_like(n)
            flat_n = n.ravel()
            num_elements = flat_n.size

            for block_start in range(0, num_elements, self.block_size):
                block_end = min(block_start + self.block_size, num_elements)
                batch_indices = list(range(block_start, block_end))

                pos_noise_batch = []
                neg_noise_batch = []

                noise_pos = deepcopy(noise)
                noise_neg = deepcopy(noise)

                noise_pos[i] = noise_pos[i].copy()
                noise_neg[i] = noise_neg[i].copy()

                for idx in batch_indices:
                    noise_pos[i].flat[idx] += self.epsilon
                    noise_neg[i].flat[idx] -= self.epsilon

                pos_noise_batch.append(noise_pos)
                neg_noise_batch.append(noise_neg)

                all_noise_batches = pos_noise_batch + neg_noise_batch
                all_samples = [
                    sample + construct_perturbation_fn(nz) for nz in all_noise_batches]

                predictions = []
                for batch_start in range(0, len(all_samples), self.batch_size):
                    batch_end = batch_start + self.batch_size
                    batch_preds = predict_fn(
                        all_samples[batch_start:batch_end])
                    predictions.extend(batch_preds)

                losses = []
                for idx, nz in enumerate(all_noise_batches):
                    perturbation = construct_perturbation_fn(nz)
                    losses.append(loss.calculate(
                        target=target_vector, predictions=predictions[idx], logits=None, noise=perturbation))

                for idx in batch_indices:
                    grad_i.flat[idx] = (losses[0] - losses[1]) / (2 * self.epsilon)

            grads_wrt_noise.append(grad_i)

        return grads_wrt_noise
