from typing import Dict, List
from . import Tracker

from adversarial_lab.core.losses import Loss
from adversarial_lab.core.penalties import Penalty

class LossTracker(Tracker):
    _columns = {"epoch_losses": "json", 
               "epoch_losses_by_batch": "json"}

    def __init__(self,
                 track_batch: bool = True,
                 track_epoch: bool = True,
                 track_loss: bool = True,
                 track_penalties: bool = True
                 ) -> None:
        super().__init__(track_batch=track_batch, track_epoch=track_epoch)
        self.track_loss = track_loss
        self.track_penalties = track_penalties

    def post_batch(self,
                   *args,
                   **kwargs
                   ) -> None:
        loss: Loss = kwargs["loss"]
        
        if not self.track_batch:
            return

        if loss is not None and self.track_loss:
            self.epoch_losses_by_batch[f"Loss {repr(loss)}"].append(loss.value)

        if loss is not None and self.track_penalties:
            for i, penalty in enumerate(loss.penalties):
                penalty_name = f"Penalty {i+1}_{repr(penalty)}"
                if penalty_name not in self.epoch_losses_by_batch:
                    self.epoch_losses_by_batch[f"{i+1}_{repr(penalty)}"] = []
                self.epoch_losses_by_batch[penalty_name] = penalty.value

        self.epoch_losses_by_batch["Total Loss"] = sum(self.epoch_losses_by_batch.values())
        
    def post_epoch(self,
                   *args,
                   **kwargs
                   ) -> None:
        loss: Loss = kwargs["loss"]

        if not self.track_epoch:
            return

        if loss is not None and self.track_loss:
            self.epoch_losses[f"Loss {repr(loss)}"] = loss.value

        if loss is not None and self.track_penalties:
            for i, penalty in enumerate(loss.penalties):
                penalty_name = f"Penalty {i+1}:{repr(penalty)}"
                self.epoch_losses[penalty_name] = penalty.value

        self.epoch_losses["Total Loss"] = sum(self.epoch_losses.values())

    def serialize(self) -> Dict:
        data = {}

        if self.track_batch:
            data["epoch_losses_by_batch"] = self.epoch_losses_by_batch    
        
        if self.track_epoch:
            data["epoch_losses"] = self.epoch_losses

        return data
    
    def reset_values(self) -> None:
        self.epoch_losses = {}
        self.epoch_losses_by_batch: Dict[str, List] = {}