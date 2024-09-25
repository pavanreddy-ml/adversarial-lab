import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import unittest
from typing import List, Type, Tuple, Dict

from adversary.models.losses import CategoricalCrossEntropy

import torch
import tensorflow as tf
import numpy as np

# Loss, dimensions_dict, IsLossAlwaysPositive
LOSS_FUNCTIONS: List[Tuple[Type, Dict[str, Tuple[int, ...]], bool]] = [
    (CategoricalCrossEntropy, {'output_dim': (32, 10), 'target_dim': (32, 10)}, True),
    # (BinaryCrossEntropy, {'output_dim': (32, 1), 'target_dim': (32, 1)}, True),
    # (MeanSquaredError, {'output_dim': (32, 1), 'target_dim': (32, 1)}, False),  # Example of a signed loss
]

class RandomDataGenerator:
    @staticmethod
    def generate_data(dim: Tuple[int, ...], framework: str):
        data = np.random.rand(*dim).astype(np.float32)
        if framework == "torch":
            return torch.tensor(data)
        elif framework == "tf":
            return tf.constant(data)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    @staticmethod
    def generate_target(dim: Tuple[int, ...], framework: str, binary: bool = False):
        if binary:
            data = np.random.randint(0, 2, size=dim).astype(np.float32)
        else:
            data = np.eye(dim[1])[np.random.choice(dim[1], dim[0])].astype(np.float32)
        if framework == "torch":
            return torch.tensor(data)
        elif framework == "tf":
            return tf.constant(data)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        self.torch_losses = []
        self.tf_losses = []
        for loss_class, dims, always_positive in LOSS_FUNCTIONS:
            self.torch_losses.append((loss_class(framework="torch"), dims, always_positive))
            self.tf_losses.append((loss_class(framework="tf"), dims, always_positive))

    def test_initialization(self):
        for loss, _, _ in self.torch_losses:
            self.assertEqual(loss.framework, "torch")
        for loss, _, _ in self.tf_losses:
            self.assertEqual(loss.framework, "tf")

    def test_invalid_framework(self):
        for loss_class, _, _ in LOSS_FUNCTIONS:
            with self.assertRaises(ValueError):
                loss_class(framework="invalid")

    def test_torch_calculation(self):
        for loss, dims, always_positive in self.torch_losses:
            with self.subTest(loss=loss.__class__.__name__):
                output = RandomDataGenerator.generate_data(dims['output_dim'], "torch")
                target = RandomDataGenerator.generate_target(dims['target_dim'], "torch", 
                                                             binary=isinstance(loss, BinaryCrossEntropy))
                result = loss.calculate(output, target)
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.dim(), 0)  # Scalar output
                if always_positive:
                    self.assertGreater(result.item(), 0)  # Loss should be positive
                else:
                    self.assertIsInstance(result.item(), float)  # Loss can be positive or negative

    def test_tf_calculation(self):
        for loss, dims, always_positive in self.tf_losses:
            with self.subTest(loss=loss.__class__.__name__):
                output = RandomDataGenerator.generate_data(dims['output_dim'], "tf")
                target = RandomDataGenerator.generate_target(dims['target_dim'], "tf", 
                                                             binary=isinstance(loss, BinaryCrossEntropy))
                result = loss.calculate(output, target)
                self.assertIsInstance(result, tf.Tensor)
                self.assertEqual(result.shape, ())  # Scalar output
                if always_positive:
                    self.assertGreater(result.numpy(), 0)  # Loss should be positive
                else:
                    self.assertIsInstance(result.numpy(), float)  # Loss can be positive or negative

    def test_torch_vs_tf_consistency(self):
        for (torch_loss, torch_dims, _), (tf_loss, tf_dims, _) in zip(self.torch_losses, self.tf_losses):
            with self.subTest(loss=torch_loss.__class__.__name__):
                torch_output = RandomDataGenerator.generate_data(torch_dims['output_dim'], "torch")
                torch_target = RandomDataGenerator.generate_target(torch_dims['target_dim'], "torch", 
                                                                   binary=isinstance(torch_loss, BinaryCrossEntropy))
                tf_output = RandomDataGenerator.generate_data(tf_dims['output_dim'], "tf")
                tf_target = RandomDataGenerator.generate_target(tf_dims['target_dim'], "tf", 
                                                                binary=isinstance(tf_loss, BinaryCrossEntropy))

                torch_result = torch_loss.calculate(torch_output, torch_target)
                tf_result = tf_loss.calculate(tf_output, tf_target)

                self.assertAlmostEqual(torch_result.item(), tf_result.numpy(), places=5)

if __name__ == '__main__':
    unittest.main()