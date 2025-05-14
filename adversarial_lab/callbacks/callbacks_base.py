from abc import ABC, abstractmethod
from typing import Literal, Optional, Dict, Any

import numpy as np

from adversarial_lab.core.losses import Loss


class Callback(ABC):
    valid_triggers = [
        "misclassification",
        "confidence_reduction",
        "loss_convergence",
        "prediction_convergence",
    ]

    def __init__(self,
                 trigger: Literal["misclassification", "confidence_reduction", "loss_convergence", "prediction_convergence"],
                 blocking=True,
                 patience: int = 1,
                 epsilon: float = 1e-3,
                 target_class: Optional[int] = None,
                 confidence: Optional[float] = None,
                 max_triggers: int = int(1e6),
                 reset_on_trigger: bool = True,
                 *args,
                 **kwargs):
        self.enabled = True
        self._original_blocking_state = blocking
        self.blocking = blocking

        if trigger not in self.valid_triggers:
            raise ValueError(f"Invalid trigger: {trigger}. Must be one of {', '.join(repr(i) for i in self.valid_triggers)}.")

        self.trigger = trigger
        self.max_triggers = max_triggers
        self.trigger_counter = 0
        self.reset_on_trigger = reset_on_trigger

        self.patience = patience
        self.patience_counter = 0
        self.epsilon = epsilon

        self.target_class = target_class
        self.confidence = confidence

        if trigger == "prediction_convergence":
            self.value_tracker = None
        else:
            self.value_tracker = float('inf')

        self.prediction_convergence_mode = kwargs.get("prediction_convergence_mode", "l1")

    @abstractmethod
    def on_epoch_end(self,
                     predictions: np.ndarray,
                     true_class: int,
                     target_class: int,
                     loss: Loss,
                     *args,
                     **kwargs) -> Optional[Dict[str, Any]]:
        pass

    def is_triggered(self,
                     predictions: np.ndarray,
                     true_class: int,
                     target_class: int,
                     loss: Loss) -> bool:
        if self.trigger == "misclassification":
            return self._on_misclassification(predictions, true_class)
        elif self.trigger == "loss_convergence":
            return self._on_loss_convergence(loss)
        elif self.trigger == "prediction_convergence":
            return self._on_prediction_convergence(predictions)
        elif self.trigger == "confidence_reduction":
            return self._on_confidence_reduction(predictions, true_class)
        else:
            raise ValueError(f"Invalid trigger: {self.trigger}. Must be one of {', '.join(repr(i) for i in self.valid_triggers)}.")

    def _on_misclassification(self,
                              predictions: np.ndarray,
                              true_class: int,
                              *args,
                              **kwargs):
        pred_class = np.argmax(predictions)
        pred_confidence = np.max(predictions)

        if self.target_class is not None:
            misclassified = pred_class == self.target_class
        else:
            misclassified = pred_class != true_class

        if misclassified:
            if self.confidence is not None and pred_confidence < self.confidence:
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        else:
            self.patience_counter = 0

        if self.patience_counter >= self.patience:
            self.trigger_counter += 1
            if self.reset_on_trigger:
                self.reset()
            if self.trigger_counter >= self.max_triggers:
                self.enabled = False
            return True

        return False

    def _on_confidence_reduction(self,
                                 predictions: np.ndarray,
                                 true_class: int,
                                 *args,
                                 **kwargs):
        if self.confidence is None: 
            self.confidence = 0.5
            
        if np.max(predictions) < self.confidence:
            self.patience_counter += 1
        else:
            self.patience_counter = 0

        if self.patience_counter >= self.patience:
            self.trigger_counter += 1
            if self.reset_on_trigger:
                self.reset()
            if self.trigger_counter >= self.max_triggers:
                self.enabled = False
            return True
        return False

    def _on_loss_convergence(self,
                             loss: Loss,
                             *args,
                             **kwargs):
        loss_value = loss.get_total_loss()
        if abs(loss_value - self.value_tracker) < self.epsilon:
            self.patience_counter += 1
        else:
            self.patience_counter = 0

        self.value_tracker = loss_value
        if self.patience_counter >= self.patience:
            self.trigger_counter += 1
            if self.reset_on_trigger:
                self.reset()
            if self.trigger_counter >= self.max_triggers:
                self.enabled = False
            return True

        return False

    def _on_prediction_convergence(self,
                                   predictions: np.ndarray,
                                   *args, **kwargs):
        if self.value_tracker is None:
            self.value_tracker = predictions.copy()
            return False

        if self.prediction_convergence_mode == "l1":
            diff = np.sum(np.abs(predictions - self.value_tracker)) / predictions.size
        elif self.prediction_convergence_mode == "l2":
            diff = np.sqrt(np.sum((predictions - self.value_tracker) ** 2)) / predictions.size
        else:
            raise ValueError(f"Invalid prediction convergence mode: {self.prediction_convergence_mode}. Must be 'l1' or 'l2'.")

        if diff < self.epsilon:
            self.patience_counter += 1
        else:
            self.patience_counter = 0

        self.value_tracker = predictions.copy()

        if self.patience_counter >= self.patience:
            self.trigger_counter += 1
            if self.reset_on_trigger:
                self.reset()
            if self.trigger_counter >= self.max_triggers:
                self.enabled = False
            return True

        return False

    def reset(self):
        self.patience_counter = 0
        self.value_tracker = float('inf') if self.trigger != "prediction_convergence" else None

    def reinitialize(self):
        self.enabled = True
        self.blocking = self._original_blocking_state
        self.trigger_counter = 0
        self.reset()
