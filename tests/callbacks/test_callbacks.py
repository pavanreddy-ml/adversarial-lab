import pytest
import numpy as np
import tensorflow as tf
import torch

from adversarial_lab.callbacks import *
from adversarial_lab.core.losses import MeanSquaredError
from adversarial_lab.core.optimizers import Adam

class DummyLoss(MeanSquaredError):
    def get_total_loss(self):
        return self.value if self.value is not None else 0.0

@pytest.mark.parametrize("trigger", Callback.valid_triggers)
def test_early_stopping_trigger(trigger):
    cb = EarlyStopping(trigger=trigger, patience=1, epsilon=0.1)
    predictions = np.array([0.1, 0.8, 0.1])
    true_class = 0
    target_class = 1
    loss = DummyLoss()
    loss.set_framework("tf")
    loss.value = 0.5
    cb.value_tracker = 0.5 if trigger != "prediction_convergence" else predictions.copy()

    for _ in range(2):
        out = cb.on_epoch_end(predictions, true_class, target_class, loss)

    assert isinstance(out, dict) or out is None
    if out:
        assert "stop_attack" in out

@pytest.mark.parametrize("trigger", Callback.valid_triggers)
def test_change_params_trigger(trigger):
    params = {
        "optimizer": {"lr": "+0.1"},
        "loss": {"scale": "*0.9"},
        "penalties": [{"alpha": "-0.1"}]
    }
    cb = ChangeParams(trigger=trigger, patience=1, epsilon=0.1, params=params)
    predictions = np.array([0.1, 0.8, 0.1])
    true_class = 0
    target_class = 1
    loss = DummyLoss()
    loss.set_framework("tf")
    loss.value = 0.5
    cb.value_tracker = 0.5 if trigger != "prediction_convergence" else predictions.copy()

    for _ in range(2):
        out = cb.on_epoch_end(predictions, true_class, target_class, loss)

    assert isinstance(out, dict) or out is None
    if out:
        assert "change_params" in out
        assert out["change_params"] == params

def test_invalid_trigger():
    with pytest.raises(ValueError, match="Invalid trigger:"):
        EarlyStopping(trigger="invalid")

def test_change_params_validation():
    with pytest.raises(ValueError, match="params must be a dictionary"):
        ChangeParams(params="not_a_dict", trigger="loss_convergence")

    with pytest.raises(ValueError, match="Unsupported top-level key"):
        ChangeParams(params={"unknown": {}}, trigger="loss_convergence")

    with pytest.raises(ValueError, match="Each item in 'penalties' must be a dictionary"):
        ChangeParams(params={"penalties": ["not_dict"]}, trigger="loss_convergence")

    with pytest.raises(ValueError, match="must start with"):
        ChangeParams(params={"loss": {"scale": "wrong"}}, trigger="loss_convergence")

    with pytest.raises(ValueError, match="must be int or float"):
        ChangeParams(params={"penalties": [{"alpha": ["a", 1]}]}, trigger="loss_convergence")

    with pytest.raises(ValueError, match="Can only be str or list"):
        ChangeParams(params={"loss": {"scale": 3.14}}, trigger="loss_convergence")

    with pytest.raises(ValueError, match="must be a string"):
        ChangeParams(params={"loss": {1: "+0.1"}}, trigger="loss_convergence")

def test_prediction_convergence_mode_variants():
    predictions = np.array([0.1, 0.8, 0.1])
    loss = DummyLoss()
    loss.set_framework("tf")
    loss.value = 0.5

    # Case 1: Not passed (should default to 'l1')
    cb_default = EarlyStopping(trigger="prediction_convergence", patience=1, epsilon=0.1)
    cb_default.on_epoch_end(predictions, 0, 1, loss)
    cb_default.on_epoch_end(predictions, 0, 1, loss)
    assert cb_default.prediction_convergence_mode == "l1"

    # Case 2: Correct value passed
    cb_l2 = EarlyStopping(trigger="prediction_convergence", patience=1, epsilon=0.1, prediction_convergence_mode="l2")
    cb_l2.on_epoch_end(predictions, 0, 1, loss)
    cb_l2.on_epoch_end(predictions, 0, 1, loss)
    assert cb_l2.prediction_convergence_mode == "l2"

    # Case 3: Incorrect value passed
    cb_invalid = EarlyStopping(trigger="prediction_convergence", patience=1, epsilon=0.1, prediction_convergence_mode="invalid")
    cb_invalid.on_epoch_end(predictions, 0, 1, loss)  # Initializes value_tracker
    with pytest.raises(ValueError, match="Invalid prediction convergence mode"):
        cb_invalid.is_triggered(predictions, 0, 1, loss)

def test_reset_behavior():
    cb = EarlyStopping(trigger="loss_convergence", patience=1, epsilon=0.1)
    cb.patience_counter = 5
    cb.value_tracker = 10
    cb.reset()
    assert cb.patience_counter == 0
    assert cb.value_tracker == float('inf')

@pytest.mark.parametrize("preds,true_class,target_class,confidence,expected", [
    (np.array([0.1, 0.9]), 0, None, None, True),          # misclassified (no target_class, no confidence)
    (np.array([0.1, 0.9]), 0, None, 0.95, False),         # misclassified, low confidence
    (np.array([0.1, 0.96]), 0, None, 0.95, True),         # misclassified, high confidence
    (np.array([0.9, 0.1]), 0, 1, 0.95, False),            # not targeted
    (np.array([0.1, 0.95]), 0, 1, 0.90, True),            # targeted and confident
])
def test_combined_misclassification_logic(preds, true_class, target_class, confidence, expected):
    cb = EarlyStopping(trigger="misclassification", patience=1, confidence=confidence, target_class=target_class)
    loss = DummyLoss()
    loss.set_framework("tf")
    loss.value = 0.1
    cb.value_tracker = 0.1

    result = cb.is_triggered(preds, true_class, target_class, loss)
    assert result == expected


def test_reinitialize_restores_state():
    cb = EarlyStopping(trigger="confidence_reduction", patience=2, confidence=0.5)
    cb.enabled = False
    cb.blocking = False
    cb.trigger_counter = 999
    cb.patience_counter = 3
    cb.value_tracker = 123

    cb.reinitialize()

    assert cb.enabled is True
    assert cb.blocking == cb._original_blocking_state
    assert cb.trigger_counter == 0
    assert cb.patience_counter == 0
    assert cb.value_tracker == float('inf')