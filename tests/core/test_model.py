import pytest
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from adversarial_lab.core.losses import Loss
except ImportError:
    tf = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None

from adversarial_lab.core.model import ALModel, ALModelTF, ALModelTorch


@pytest.mark.skipif(torch is None, reason="Torch not installed.")
def test_framework_detection_torch():
    class SimpleTorchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    model = SimpleTorchModel()
    al_model = ALModel(model)

    assert isinstance(al_model, ALModelTorch)
    assert al_model.framework == "torch"


@pytest.mark.skipif(tf is None, reason="TensorFlow not installed.")
def test_framework_detection_tf():
    model = models.Sequential([
        layers.Dense(10, input_shape=(10,)),
        layers.Dense(5)
    ])

    al_model = ALModel(model)

    assert isinstance(al_model, ALModelTF)
    assert al_model.framework == "tf"

@pytest.mark.skipif(torch is None, reason="Torch not installed.")
def test_torch_model_info():
    class SimpleTorchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 2)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    model = SimpleTorchModel()
    al_model = ALModelTorch(model)
    info = al_model.get_info(model)

    assert info["num_params"] > 0
    assert len(info["layers_info"]) == 2
    assert "Linear" in [layer["layer_name"] for layer in info["layers_info"]]


@pytest.mark.skipif(tf is None, reason="TensorFlow not installed.")
def test_tf_model_info():
    model = models.Sequential([
        layers.Dense(10, input_shape=(10,)),
        layers.Dense(5, activation="relu"),
        layers.Dense(2)
    ])

    al_model = ALModelTF(model)
    info = al_model.get_info(model)

    assert info["num_params"] > 0
    assert len(info["layers_info"]) == 3
    assert "Dense" in [layer["layer_type"] for layer in info["layers_info"]]

@pytest.mark.skipif(torch is None, reason="Torch not installed.")
def test_torch_model_predict_forward():
    class SimpleTorchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc1(x)

    model = SimpleTorchModel()
    al_model = ALModelTorch(model)
    
    dummy_input = torch.randn(1, 10)
    
    with pytest.raises(NotImplementedError):
        al_model.predict(dummy_input)

    with pytest.raises(NotImplementedError):
        al_model.forward(dummy_input)


@pytest.mark.skipif(tf is None, reason="TensorFlow not installed.")
def test_tf_model_predict_forward():
    model = models.Sequential([
        layers.Dense(10, input_shape=(10,)),
        layers.Dense(5, activation="relu"),
        layers.Dense(2)
    ])

    al_model = ALModelTF(model)
    dummy_input = np.random.rand(1, 10).astype(np.float32)

    pred = al_model.predict(dummy_input)
    fwd = al_model.forward(dummy_input)

    assert pred.shape == (1, 2)
    assert fwd.shape == (1, 2)

@pytest.mark.skipif(tf is None, reason="TensorFlow not installed.")
def test_tf_model_compute_jacobian_flag():
    model = models.Sequential([
        layers.Dense(10, input_shape=(10,)),
        layers.Dense(5, activation="relu"),
        layers.Dense(2)
    ])

    sample = tf.random.normal((1, 10))
    noise_meta = [tf.Variable(tf.random.normal((1, 10)))]

    def construct_perturbation_fn(noise_meta):
        return sample + noise_meta[0]

    target_vector = tf.random.normal((1, 2))

    class TestLoss(Loss):
        def calculate(self, target, predictions, logits, noise):
            return tf.reduce_mean((target - predictions) ** 2)

    loss = TestLoss()

    # With compute_jacobian = True
    al_model = ALModelTF(model)
    al_model.set_compute_jacobian(True)
    grad_loss, grad_logits = al_model.calculate_gradients(sample, noise_meta, construct_perturbation_fn, target_vector, loss)
    assert grad_loss is not None
    assert grad_logits is not None

    # With compute_jacobian = False
    al_model = ALModelTF(model)
    al_model.set_compute_jacobian(False)
    grad_loss, grad_logits = al_model.calculate_gradients(sample, noise_meta, construct_perturbation_fn, target_vector, loss)
    assert grad_loss is not None
    assert grad_logits is None


@pytest.mark.skipif(torch is None, reason="Torch not installed.")
def test_torch_model_calculate_gradients():
    class SimpleTorchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc1(x)

    model = SimpleTorchModel()
    al_model = ALModelTorch(model)

    with pytest.raises(NotImplementedError):
        al_model.calculate_gradients(None, None, None, None, None)
