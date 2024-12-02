from abc import ABC, ABCMeta, abstractmethod
import sys
import torch
import tensorflow as tf
from typing import Dict, Any, List, Tuple, TypeVar, Generic, Literal

from adversarial_lab.core.losses import Loss
from adversarial_lab.core.noise_generators import NoiseGenerator
from adversarial_lab.exceptions import IndifferentiabilityError

class ALModelMeta(ABCMeta):
    def __call__(cls, model, *args, **kwargs):

        if isinstance(model, torch.nn.Module):
            framework = "torch"
        elif isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
            framework = "tf"
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        # framework = None

        # if model is not None:
        #     if hasattr(model, "parameters") and callable(getattr(model, "parameters", None)):
        #         try:
        #             import torch
        #             if isinstance(model, torch.nn.Module):
        #                 framework = "torch"
        #         except ImportError:
        #             pass

        #     if framework is None and hasattr(model, "layers") and hasattr(model, "count_params"):
        #         try:
        #             import tensorflow as tf
        #             if isinstance(model, tf.keras.Model):
        #                 framework = "tf"
        #         except ImportError:
        #             pass

        # if framework is None:
        #     raise ImportError(
        #         "Unable to determine framework. Either PyTorch or TensorFlow must be installed, "
        #         "and the model must be an instance of torch.nn.Module or tf.keras.Model."
        #     )

        if framework == "torch":
            specific_class = ALModelTorch
        elif framework == "tf":
            specific_class = ALModelTF
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
        instance = specific_class(model, *args, **kwargs)
        return instance

class ALModelBase(ABC):
    def __init__(self, model: str) -> None:
        self.model = model
        self.model_info = self.get_info(model)

    @abstractmethod
    def get_info(self, model) -> Dict[str, Any]:
        pass

    @abstractmethod
    def calculate_gradients(self,
                            sample: Any,
                            noise: List[Any],
                            noise_generator: NoiseGenerator,
                            target_vector: Any,
                            loss: Loss
                            ) -> Tuple[List[Any], float]:
        pass

    @abstractmethod
    def predict(self, x: Any) -> Any:
        pass

    @abstractmethod
    def forward(self, x: Any) -> Any:
        pass

class ALModelTorch(ALModelBase):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.framework = "torch"

    def get_info(self, 
                 model: torch.nn.Module
                 ) -> Dict[str, Any]:
        num_params = sum(p.numel() for p in model.parameters())

        layers_info = []
        total_units = 0
        for layer in model.children():
            layer_info = {
                "layer_name": layer.__class__.__name__,
                "layer_type": type(layer).__name__,
            }

            if hasattr(layer, 'weight'):
                layer_info["units"] = layer.weight.shape[0]
                total_units += layer.weight.shape[0]

            if isinstance(layer, torch.nn.ReLU):
                layer_info["activation"] = "ReLU"
            elif isinstance(layer, torch.nn.Sigmoid):
                layer_info["activation"] = "Sigmoid"
            elif isinstance(layer, torch.nn.Tanh):
                layer_info["activation"] = "Tanh"
            elif isinstance(layer, torch.nn.LeakyReLU):
                layer_info["activation"] = "LeakyReLU"
            elif isinstance(layer, torch.nn.Softmax):
                layer_info["activation"] = "Softmax"

            layers_info.append(layer_info)

        input_shape = None
        output_shape = None
        try:
            example_input = torch.randn(1, *model.input_shape)
            output = model(example_input)
            input_shape = example_input.shape
            output_shape = output.shape
        except AttributeError:
            pass

        param_info = [
            {"name": name, "shape": param.shape, "requires_grad": param.requires_grad}
            for name, param in model.named_parameters()
        ]

        model_info = {
            "num_params": num_params,
            "total_units": total_units,
            "layers_info": layers_info,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "params": param_info,
        }
        return model_info
    
    def calculate_gradients(self,
                            sample: Any,
                            noise: List[Any],
                            noise_generator: NoiseGenerator,
                            target_vector: Any,
                            loss: Loss
                            ) -> Tuple[List[Any], float]:
        raise NotImplementedError()
    
    def predict(self, 
                x: Any
                ) -> Any:
        raise NotImplementedError()
    
    def forward(self,
                x: Any
                ) -> Any:
        raise NotImplementedError()
                            

class ALModelTF(ALModelBase):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.framework = "tf"

    def get_info(self, model: tf.keras.Model) -> Dict[str, Any]:
        num_params = model.count_params()

        layers_info = []
        total_units = 0
        for layer in model.layers:
            layer_info = {
                "layer_name": layer.name,
                "layer_type": type(layer).__name__,
                "trainable": layer.trainable
            }

            if hasattr(layer, 'units'):
                layer_info["units"] = layer.units
                total_units += layer.units

            if hasattr(layer, 'kernel_size'):
                layer_info["kernel_size"] = layer.kernel_size

            if hasattr(layer, 'activation'):
                if layer.activation == tf.keras.activations.relu:
                    layer_info["activation"] = "ReLU"
                elif layer.activation == tf.keras.activations.sigmoid:
                    layer_info["activation"] = "Sigmoid"
                elif layer.activation == tf.keras.activations.tanh:
                    layer_info["activation"] = "Tanh"
                elif layer.activation == tf.keras.activations.softmax:
                    layer_info["activation"] = "Softmax"
                else:
                    layer_info["activation"] = layer.activation.__name__

            layers_info.append(layer_info)

        input_shape = model.input_shape if hasattr(model, 'input_shape') else None
        output_shape = model.output_shape if hasattr(model, 'output_shape') else None

        param_info = [
            {"name": weight.name, "shape": weight.shape, "trainable": weight.trainable}
            for weight in model.weights
        ]

        model_info = {
            "num_params": num_params,
            "total_units": total_units,
            "layers_info": layers_info,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "params": param_info,
        }
        return model_info
    
    def calculate_gradients(self,
          sample: tf.Tensor,
          noise: List[tf.Tensor],
          noise_generator: NoiseGenerator,
          target_vector: tf.Tensor,
          loss: Loss
          ) -> Tuple[List[tf.Tensor], float]:
        """
        Calculate gradients and return them along with the scalar loss.

        Parameters:
        ----------
        sample : tf.Tensor
            The input sample being perturbed.
        noise : tf.Tensor
            The noise added to the input sample.
        noise_generator : NoiseGenerator
            The generator that applies noise to the input sample.
        target_vector : tf.Tensor
            The target labels for the sample.
        loss : Loss
            The loss function used to compute gradients with respect to model outputs and targets.

        Returns:
        -------
        Tuple[tf.Tensor, float]
            A tuple containing the gradients (tf.Tensor) and the scalar loss value (float).
        """
        
        with tf.GradientTape() as tape:
            noise = [tf.Variable(n, trainable=True) for n in noise]
            for n in noise:
                tape.watch(n)
            input = noise_generator.apply_noise(sample, noise)
            outputs = self.model(input)
            if len(target_vector.shape) == 1:
                target_vector = tf.expand_dims(target_vector, axis=0)
            error = loss.calculate(outputs, target_vector)
  
        gradients = tape.gradient(error, noise)
        if gradients is None:
            raise IndifferentiabilityError()
        
        gradients = [tf.squeeze(grad, axis=0) if grad.shape[0] == 1 and len(n.shape) < len(grad.shape) else grad 
                    for grad, n in zip(gradients, noise)]
        return gradients, tf.reduce_mean(error).numpy().item()
    
    def predict(self, 
                x: tf.Tensor
                ) -> tf.Tensor:
        return self.model.predict(x, verbose=0)
    
    def forward(self, 
                x: tf.Tensor
                ) -> tf.Tensor:
        return self.model(x)


class ALModel(ALModelBase, metaclass=ALModelMeta):
    def __init__(self, 
                 model, 
                 *args, 
                 **kwargs
                 ) -> None:
        self.model = model
        self.model_info = self.get_info(model)