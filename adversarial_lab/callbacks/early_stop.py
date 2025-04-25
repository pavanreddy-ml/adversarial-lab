from . import Callback

import numpy as np
from adversarial_lab.core.losses import Loss

from typing import Literal, Dict, Optional


class EarlyStopping(Callback):
    def __init__(self, 
                 trigger: Literal["misclassification", "targeted_misclassification", "confidence_reduction", "loss_convergence", "prediction_convergence"],
                 patience: int = 1, 
                 epsilon: float = 1e-3,
                 *args,
                 **kwargs
                 ):
        super().__init__(trigger=trigger, 
                         blocking=True,
                         patience=patience,
                         epsilon=epsilon,
                         max_triggers=1,
                         *args,
                         **kwargs)

    def on_epoch_end(self,
                     predictions: np.ndarray,
                     true_class: int,
                     target_class: int,
                     loss: Loss) -> Optional[Dict[str, bool]]:
        if self.is_triggered(predictions, true_class, target_class, loss):
            return {
                "stop_attack": True,
            }
