from typing import Dict, List, Literal
from . import Tracker

import numpy as np


class NoiseStatisticsTracker(Tracker):
    _columns = {
        "epoch_predictions": "json",
        "epoch_predictions_by_batch": "json"
    }

    def __init__(self,
                 track_batch: bool = True,
                 track_epoch: bool = True,
                 track_mean: bool = True,
                 track_median: bool = True,
                 track_std: bool = True,
                 track_min: bool = True,
                 track_max: bool = True,
                 round_to: int = 8
                 ) -> None:
        super().__init__(track_batch=track_batch, track_epoch=track_epoch)
        self.track_mean = track_mean
        self.track_median = track_median
        self.track_std = track_std
        self.track_min = track_min
        self.track_max = track_max

        self.round_to = round_to

    def post_batch(self,
                   *args,
                   **kwargs
                   ) -> None:
        pass

    def post_epoch(self,
                   *args,
                   **kwargs
                   ) -> None:
        if not self.track_epoch:
            return

        noise_raw_image = kwargs.get("noise_raw", None)
        noise_preprocessed_image = kwargs.get("noise_preprocessed_image", None)

        if noise_raw_image and noise_preprocessed_image:
            return
        
        noise = noise_raw_image if noise_raw_image is not None else noise_preprocessed_image



    def _initialize_index(self,
                          predictions: np.ndarray
                          ) -> None:
        pass

    def serialize(self) -> Dict:
        data = {}

        return data

    def reset_values(self) -> None:
        self.data = {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
