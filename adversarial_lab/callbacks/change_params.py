from . import Callback

import numpy as np
from adversarial_lab.core.losses import Loss

from typing import Dict, Any, Literal, Optional


class ChangeParams(Callback):
    def __init__(self,
                 params,
                 trigger: Literal["misclassification", "targeted_misclassification", "confidence_reduction", "loss_convergence", "prediction_convergence"],
                 blocking=True,
                 patience: int = 1,
                 epsilon: float = 1e-3,
                 target_class: Optional[int] = None,
                 confidence: Optional[float] = None,
                 max_triggers: int = int(1e6),
                 *args,
                 **kwargs
                 ):
        super().__init__(trigger=trigger,
                         blocking=blocking,
                         patience=patience,
                         epsilon=epsilon,
                         target_class=target_class,
                         confidence=confidence,
                         max_triggers=max_triggers,
                         *args,
                         **kwargs)
        self._validate_params(params)
        self.params = params

    def on_epoch_end(self,
                     predictions: np.ndarray,
                     true_class: int,
                     target_class: int,
                     loss: Loss
                     ) -> Dict[str, Any]:
        if self.is_triggered(predictions, true_class, target_class, loss):
            return {
                "change_params": self.params
            }

    def _validate_params(self, params):
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary")

        allowed_keys = {"optimizer", "loss", "penalties"}
        for key, value in params.items():
            if key not in allowed_keys:
                raise ValueError(
                    f"Unsupported top-level key '{key}'. Allowed keys are: {allowed_keys}")

            if key == "penalties":
                if not isinstance(value, list):
                    raise ValueError(
                        "'penalties' must be a list of dictionaries")
                for item in value:
                    if not isinstance(item, dict):
                        raise ValueError(
                            "Each item in 'penalties' must be a dictionary")
                    for subkey, subval in item.items():
                        self._validate_param_entry(
                            f"{key}.{subkey}", subkey, subval)
            else:
                if not isinstance(value, dict):
                    raise ValueError(f"Value for '{key}' must be a dictionary")
                for subkey, subval in value.items():
                    self._validate_param_entry(
                        f"{key}.{subkey}", subkey, subval)

    def _validate_param_entry(self, full_key, subkey, subval):
        if not isinstance(subkey, str):
            raise ValueError(
                f"Key '{full_key}' must be a string. Got {type(subkey)}")

        if isinstance(subval, str):
            if not (subval[0] in '+-*/' and self._is_number(subval[1:])):
                raise ValueError(
                    f"String value '{subval}' in '{full_key}' must start with +, -, *, or / followed by a number")
        elif isinstance(subval, list):
            if not all(isinstance(x, (int, float)) for x in subval):
                raise ValueError(
                    f"All elements in list for '{full_key}' must be int or float")
        else:
            raise ValueError(
                f"Invalid value type for '{full_key}': {type(subval)}. Can only be str or list.")

    def _is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def apply_changes(self,
                      optimizer: Any,
                      loss: Loss) -> None:
        pass
