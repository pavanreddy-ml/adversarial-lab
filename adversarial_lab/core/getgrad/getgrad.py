from abc import abstractmethod, ABC, ABCMeta
from typing import Literal, List, Union, Tuple

import torch
import tensorflow as tf

from adversarial_lab.core.losses import Loss
from adversarial_lab.core.getgrad import GetGradsBase
from adversarial_lab.core.noise_generators import NoiseGenerator


class GetGrads(GetGradsBase):
    """
    A class to compute gradients for adversarial attacks based on the framework used (TensorFlow or PyTorch).
    This class extends the `GetGradsBase` and provides implementations to calculate gradients in TensorFlow.
    
    The class supports TensorFlow and PyTorch models. For TensorFlow, the gradients are calculated using
    TensorFlow's `GradientTape`. For PyTorch, the method `torch_op` needs to be implemented for gradient computation.
    
    Parameters
    ----------
    framework : Literal["torch", "tf"]
        Specifies the framework to use for gradient computation. 
        Should be either "torch" for PyTorch or "tf" for TensorFlow.
    loss : Loss
        The loss function used to compute gradients with respect to model outputs and targets.
    
    Methods
    -------
    calculate(model, sample, noise, noise_generator, targets)
        Calculates gradients using the framework specified during initialization.
        Delegates the computation to the `torch_op` or `tf_op` based on the framework.
        
    torch_op(model, sample, noise, noise_generator, targets)
        Not implemented for PyTorch in this class. Raises NotImplementedError.
        
    tf_op(model, sample, noise, noise_generator, targets) -> Tuple[tf.Tensor, float]
        Calculates the gradients with respect to noise using TensorFlow.
        Returns a tuple containing the gradients and the scalar loss.
    """
    def __init__(self,
                 framework: Literal["torch", "tf"],
                 loss: Loss) -> None:
        super().__init__(None, loss)
        self.framework = framework
        self.loss = loss

    def calculate(self,
                  model: Union[torch.nn.Module, tf.keras.Model],
                  sample: Union[torch.Tensor, tf.Tensor],
                  noise: Union[torch.Tensor, tf.Tensor],
                  noise_generator: NoiseGenerator,
                  targets: Union[torch.Tensor, tf.Tensor]):
        return super().calculate(model, sample, noise, noise_generator, targets)

    def torch_op(self,
                 model: Union[torch.nn.Module, tf.keras.Model],
                 sample: Union[torch.Tensor, tf.Tensor],
                 noise: Union[torch.Tensor, tf.Tensor],
                 noise_generator: NoiseGenerator,
                 targets: Union[torch.Tensor, tf.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError("Not implemented for Torch")

    def tf_op(self,
          model: tf.keras.Model,
          sample: tf.Tensor,
          noise: tf.Tensor,
          noise_generator: NoiseGenerator,
          targets: tf.Tensor
          ) -> Tuple[tf.Tensor, float]:
        """
        Calculate gradients and return them along with the scalar loss.

        Parameters:
        ----------
        model : tf.keras.Model
            The TensorFlow model being attacked.
        sample : tf.Tensor
            The input sample being perturbed.
        noise : tf.Tensor
            The noise added to the input sample.
        noise_generator : NoiseGenerator
            The generator that applies noise to the input sample.
        targets : tf.Tensor
            The target labels for the sample.

        Returns:
        -------
        Tuple[tf.Tensor, float]
            A tuple containing the gradients (tf.Tensor) and the scalar loss value (float).
        """
        noise = tf.Variable(noise, trainable=True)

        with tf.GradientTape() as tape:
            tape.watch(noise)
            
            input = noise_generator.apply_noise(sample, noise)
            outputs = model(input)
            if len(targets.shape) == 1:
                targets = tf.expand_dims(targets, axis=0)
            loss = self.loss.calculate(outputs, targets)
  
        gradients = tape.gradient(loss, noise)
        if gradients.shape[0] == 1 and len(noise.shape) < len(gradients.shape):
            gradients = tf.squeeze(gradients, axis=0)

        return gradients, tf.reduce_mean(loss).numpy().item()
