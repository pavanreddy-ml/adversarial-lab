from . import ModelInfoBase, ModelInfoTensorFlow, ModelInfoTorch
import torch
import tensorflow as tf
from typing import Dict, Any

class ModelInfo(ModelInfoBase):
    def __init__(self, framework: str) -> None:
        self.framework = framework

    def get_info(self, model) -> Dict[str, Any]:
        if self.framework == "torch":
            return self.get_info_torch(model)
        elif self.framework == "tf":
            return self.get_info_tf(model)
        else:
            raise ValueError("Framework must be either 'torch' or 'tf'")

    def get_info_torch(self, model: torch.nn.Module) -> Dict[str, Any]:
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

    def get_info_tf(self, model: tf.keras.Model) -> Dict[str, Any]:
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
