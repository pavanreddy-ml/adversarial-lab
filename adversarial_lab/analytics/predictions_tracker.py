from typing import Dict, List
from . import Tracker

import numpy as np


class PredictionsTracker(Tracker):
    columns = {
        "epoch_predictions": "json",
        "epoch_predictions_by_batch": "json"
    }

    def __init__(self,
                 track_batch: bool = True,
                 track_epoch: bool = True,
                 strategy: str = True,
                 topk: int = 5,
                 custom_indexes: List[int] = None
                 ) -> None:
        super().__init__(track_batch=track_batch, track_epoch=track_epoch)
    
        self.indexes = None

        if strategy == "all":
            self.strategy = "all"
        elif strategy == "topk":
            self.strategy = "topk"
            self.topk = topk
        elif strategy == "custom":
            self.indexes = np.array(custom_indexes)
            self.strategy = "custom"
        else:
            raise ValueError("Invalid strategy")

    def post_batch(self,
                   epoch_num: int,
                   *args,
                   **kwargs
                   ) -> None:
        pass

    def post_epoch(self,
                   epoch_num: int,
                   *args,
                   **kwargs
                   ) -> None:
        pass

    def serialize(self) -> Dict:
        data = {}

        if self.track_batch:
            data["epoch_losses_by_batch"] = self.epoch_losses_by_batch    
        
        if self.track_epoch:
            data["epoch_losses"] = self.epoch_losses

        return data