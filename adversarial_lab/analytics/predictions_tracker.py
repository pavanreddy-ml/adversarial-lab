from typing import Dict, List, Literal
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
                 strategy: Literal["all", "first_topk", "custom"] = "all",
                 topk: int = 5,
                 custom_indexes: List[int] = None,
                 round_to: int = 4
                 ) -> None:
        super().__init__(track_batch=track_batch, track_epoch=track_epoch)
    
        self.indexes = None
        self.strategy = "first_topk"

        if strategy == "all":           # Just for code readability
            self.strategy = "all"
        elif strategy == "first_topk":
            self.topk = topk
        elif strategy == "custom":
            if min(custom_indexes) < 0:
                raise ValueError(f"Index {min(custom_indexes)} in Custom indexes is out of bounds")
            self.custom_indexes = custom_indexes
        else:
            raise ValueError("Invalid strategy")
        
        self.round_to = round_to

    def post_batch(self,
                   epoch_num: int,
                   predictions: np.ndarray,
                   *args,
                   **kwargs
                   ) -> None:
        if not self.track_batch:
            return
        
        self._initialize_index(predictions)

        for i, value in enumerate(predictions):
            if self.indexes[i] == 1:
                if i not in self.epoch_predictions_by_batch:
                    self.epoch_predictions_by_batch[i] = []
                self.epoch_predictions_by_batch[i].append(round(float(value), self.round_to))
                
    def post_epoch(self,
                   epoch_num: int,
                   predictions: np.ndarray,
                   *args,
                   **kwargs
                   ) -> None:
        if not self.track_epoch:
            return  
        
        self._initialize_index(predictions)

        for i, value in enumerate(predictions):
            if self.indexes[i] == 1:
                self.epoch_predictions[i] = round(float(value), self.round_to)

    def _initialize_index(self, 
                          predictions: np.ndarray
                          ) -> None:
        if predictions.shape[0] is None or predictions.shape[0] == 1:
            preds = predictions[0]
        else:
            preds = predictions

        if self.indexes is None:
            if self.strategy == "all":
                self.indexes = np.ones_like(preds)
            elif self.strategy == "first_topk":
                self.indexes = np.zeros_like(preds)
                self.indexes[np.argsort(preds)[-self.topk:]] = 1
            elif self.strategy == "custom":
                if max(self.custom_indexes) >= len(preds):
                    raise ValueError(f"Index {max(self.custom_indexes)} in Custom indexes is out of bounds")
                self.indexes = np.zeros_like(preds)
                self.indexes[self.custom_indexes] = 1

            self.indexes = self.indexes.tolist()
        
    def serialize(self) -> Dict:
        data = {}

        if self.track_batch:
            data["epoch_predictions_by_batch"] = self.epoch_predictions_by_batch    
        
        if self.track_epoch:
            data["epoch_predictions"] = self.epoch_predictions

        return data
    
    def reset_values(self) -> None:
        self.epoch_predictions = {}
        self.epoch_predictions_by_batch: Dict[str, List] = {}

