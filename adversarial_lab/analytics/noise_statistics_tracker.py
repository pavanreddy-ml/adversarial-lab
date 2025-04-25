from typing import Dict, List, Literal, Union
from . import Tracker
import numpy as np
import re

TrackedStat = Literal["mean", "median", "std", "min", "max", "var", "p25", "p75", "p_custom_x" "iqr", "skew", "kurtosis"]


class NoiseStatisticsTracker(Tracker):
    _columns = {
        "epoch_predictions": "json",
        "epoch_predictions_by_batch": "json"
    }

    def __init__(self,
                 track_batch: bool = True,
                 track_epoch: bool = True,
                 tracked_stats: List[Union[TrackedStat, str]] = ["mean", "median", "std", "min", "max", "var", "p25", "p75", "iqr", "skew", "kurtosis"],
                 round_to: int = 8
                 ) -> None:
        super().__init__(track_batch=track_batch, track_epoch=track_epoch)
        self.tracked_stats = tracked_stats
        self.round_to = round_to
        self.reset_values()

    def post_batch(self, *args, **kwargs) -> None:
        pass

    def post_epoch(self, *args, **kwargs) -> None:
        if not self.track_epoch:
            return

        noise_raw_image: np.ndarray = kwargs.get("noise_raw", None)
        noise_preprocessed_image: np.ndarray = kwargs.get("noise_preprocessed_image", None)

        if noise_raw_image is None and noise_preprocessed_image is None:
            return

        noise = noise_raw_image if noise_raw_image is not None else noise_preprocessed_image
        noise = noise.flatten()

        self.data = self._get_stats(noise)

    def _get_stats(self, 
                   noise: np.ndarray
                   ) -> Dict[str, float]:
        stats = {}
        for stat in self.tracked_stats:
            if stat == "mean":
                stats["mean"] = round(float(np.mean(noise)), self.round_to)
            elif stat == "median":
                stats["median"] = round(float(np.median(noise)), self.round_to)
            elif stat == "std":
                stats["std"] = round(float(np.std(noise)), self.round_to)
            elif stat == "min":
                stats["min"] = round(float(np.min(noise)), self.round_to)
            elif stat == "max":
                stats["max"] = round(float(np.max(noise)), self.round_to)
            elif stat == "var":
                stats["var"] = round(float(np.var(noise)), self.round_to)
            elif stat == "p25":
                stats["p25"] = round(float(np.percentile(noise, 25)), self.round_to)
            elif stat == "p75":
                stats["p75"] = round(float(np.percentile(noise, 75)), self.round_to)
            elif stat == "iqr":
                q1 = np.percentile(noise, 25)
                q3 = np.percentile(noise, 75)
                stats["iqr"] = round(float(q3 - q1), self.round_to)
            elif stat == "skew":
                stats["skew"] = round(float((np.mean((noise - np.mean(noise))**3)) / np.std(noise)**3), self.round_to)
            elif stat == "kurtosis":
                stats["kurtosis"] = round(float((np.mean((noise - np.mean(noise))**4)) / np.std(noise)**4 - 3), self.round_to)
            elif stat.startswith("p_custom_"):
                try:
                    val = float(stat.split("p_custom_")[1])
                    stats[stat] = round(float(np.percentile(noise, val)), self.round_to)
                except:
                    continue

        return stats

    def serialize(self) -> Dict:
        return {k: v for k, v in self.data.items() if k in self.tracked_stats and v is not None}

    def reset_values(self) -> None:
        self.data = {stat: None for stat in self.tracked_stats}
