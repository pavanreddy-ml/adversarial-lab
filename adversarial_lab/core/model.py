from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.exceptions import IndifferentiabilityError
from adversarial_lab.core.noise_generators import NoiseGenerator
from adversarial_lab.core.losses import Loss
from adversarial_lab.core.types import TensorType, TensorVariableType, ModelType, LossType
from abc import ABC, ABCMeta, abstractmethod
from copy import deepcopy
import sys
import importlib
import numpy as np
from typing import Dict, Any, List, Tuple, Callable, TYPE_CHECKING, Optional
import warnings

tf = None
torch = None

if TYPE_CHECKING:
    import tensorflow as tf
    import torch


class ALModelMeta(ABCMeta):
    def __call__(cls, model, *args, **kwargs):
        global torch, tf
        if hasattr(model, "parameters") and callable(getattr(model, "parameters", None)):
            framework = "torch"
        elif hasattr(model, "layers") and hasattr(model, "count_params"):
            framework = "tf"
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        if framework == "torch":
            torch = importlib.import_module("torch")
            specific_class = ALModelTorch
        elif framework == "tf":
            tf = importlib.import_module("tensorflow")
            specific_class = ALModelTF

        return specific_class(model, *args, **kwargs)


class ALModelBase(ABC):
    def __init__(self,
                 model: ModelType,
                 framework: str,
                 efficient_mode: bool = False) -> None:
        self.model = deepcopy(model)
        self.model_info = self.get_info(self.model)
        self.framework = framework
        self._compute_jacobian = False
        
        
        self.efficient_mode = efficient_mode
        if self.efficient_mode:
            warnings.warn("You are using an efficient mode. This works only on hard targets. "
            "Soft targets will raise an IncomaptibilityError.")
    

    @abstractmethod
    def get_info(self, model) -> Dict[str, Any]:
        pass

    @abstractmethod
    def calculate_gradients(self,
                            sample: TensorType,
                            noise: List[TensorVariableType],
                            apply_noise_fn: Callable,
                            target_vector: TensorType,
                            loss: LossType,
                            preprocess_fn: Optional[Callable] = None
                            ) -> Tuple[TensorType, TensorType]:
        pass

    @abstractmethod
    def predict(self, x: TensorType | np.ndarray, preprocess_fn: Optional[Callable] = None) -> Any:
        pass

    @abstractmethod
    def forward(self, x: TensorType | np.ndarray) -> Any:
        pass

    def set_compute_jacobian(self,
                             compute_jacobian: bool
                             ) -> None:
        self._compute_jacobian = compute_jacobian


class ALModelTorch(ALModelBase):
    def __init__(self,
                 model: ModelType,
                 efficient_mode: bool = False) -> None:
        super().__init__(model, "torch", efficient_mode)

    def get_info(self,
                 model: ModelType
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
            {"name": name, "shape": param.shape,
                "requires_grad": param.requires_grad}
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
                            sample: TensorType,
                            noise: List[TensorVariableType],
                            apply_noise_fn: Callable,
                            target_vector: TensorType,
                            loss: LossType,
                            preprocess_fn: Optional[Callable] = None
                            ) -> Tuple[TensorType, TensorType]:
        raise NotImplementedError()

    def predict(self,
                x: TensorType | np.ndarray,
                preprocess_fn: Optional[Callable] = None
                ) -> TensorType:
        raise NotImplementedError()

    def forward(self,
                x: TensorType | np.ndarray
                ) -> TensorType:
        raise NotImplementedError()


class ALModelTF(ALModelBase):
    def __init__(self,
                 model: ModelType,
                 efficient_mode: bool = False,) -> None:
        super().__init__(model=model, framework="tf", efficient_mode=efficient_mode)
        self.act = getattr(self.model.layers[-1], "activation", None)
        self.model.layers[-1].activation = None
        self.tensor_ops = TensorOps(framework="tf")

        if self.act is None:
            self.act = lambda x: x

    def get_info(self,
                 model: ModelType) -> Dict[str, Any]:
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

        input_shape = model.input_shape if hasattr(
            model, 'input_shape') else None
        output_shape = model.output_shape if hasattr(
            model, 'output_shape') else None

        param_info = [
            {"name": weight.name, "shape": weight.shape,
                "trainable": weight.trainable}
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

    def calculate_gradients(
        self,
        sample: TensorType,
        noise: List[TensorVariableType],
        construct_perturbation_fn: Callable,
        target_vector: TensorType,
        loss: Loss,
        preprocess_fn: Optional[Callable] = None
    ) -> Tuple[TensorType, TensorType]:
        with tf.GradientTape(persistent=True) as tape:
            for n in noise:
                tape.watch(n)

            perturbation = construct_perturbation_fn(noise)

            if preprocess_fn is not None:
                input = preprocess_fn(sample + perturbation)
            else:
                input = sample + perturbation
                
            logits = self.model(input)
            preds = self.act(logits)
            loss_value = loss.calculate(
                target=target_vector, predictions=preds, logits=logits, noise=perturbation) if loss else None

        logit_grads = tape.jacobian(
            logits, noise) if self._compute_jacobian else None
        grad_wrt_loss = tape.gradient(
            loss_value, noise) if loss_value is not None else None
        return grad_wrt_loss, logit_grads

    def predict(self,
                x: TensorType | np.ndarray,
                preprocess_fn: Optional[Callable] = None
                ) -> TensorType:
        if preprocess_fn is not None:
            sample = preprocess_fn(x)
        else:
            sample = x
        return self.act(self.model(sample, training=False))

    def forward(self,
                x: TensorType | np.ndarray
                ) -> TensorType:
        return self.act(self.model(x))


class ALModel(ALModelBase, metaclass=ALModelMeta):
    def __init__(self,
                 model: ModelType,
                 framework: str,
                 efficient_mode: bool = False,
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(model, framework)
