from typing import Dict
from . import Tracker

class LossTracker(Tracker):
    columns = {"epoch_losses": "json", 
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

        self.reset_values()

    def post_batch(self,
                   epoch_num: int,
                   *args,
                   **kwargs
                   ) -> None:
        if not self.track_batch:
            return
        
        # loss = kwargs.get("loss", None)

        # if loss is not None and self.track_loss:
        #     self.epoch_losses_by_batch["loss"].append(loss.value)

        # if loss is not None and self.track_penalties:
        #     for i, penalty in enumerate(loss.penalties):
        #         penalty_name = f"{i+1}_{penalty.name}"
        #         if penalty_name not in self.epoch_losses_by_batch:
        #             self.epoch_losses_by_batch[f"{i+1}_{penalty.name}"] = []
        #         self.epoch_losses_by_batch[penalty_name] = penalty.value
        
    def post_epoch(self,
                   epoch_num: int,
                   *args,
                   **kwargs
                   ) -> None:
        if not self.track_epoch:
            return
        
        # loss = kwargs.get("loss", None)

        # if loss is not None and self.track_loss:
        #     self.epoch_losses["loss"] = loss.value

        # if loss is not None and self.track_penalties:
        #     for i, penalty in enumerate(loss.penalties):
        #         penalty_name = f"{i+1}_{penalty.name}"
        #         self.epoch_losses[penalty_name] = penalty.value

    def serialize(self) -> Dict:
        data = {}

        if self.track_batch:
            data["epoch_losses_by_batch"] = self.epoch_losses_by_batch    
        
        if self.track_epoch:
            data["epoch_losses"] = self.epoch_losses

        return data
    
    def reset_values(self) -> None:
        self.epoch_losses = {
            "loss": None,
        }

        self.epoch_losses_by_batch = {
            "loss": [],
        }