from typing import Dict
from . import Tracker

import warnings

from adversarial_lab.utils import Conversions


class NoiseTracker(Tracker):
    _columns = {
        "noise": "blob"
    }

    def __init__(self) -> None:
        super().__init__()
        self.warned = False

    def pre_attack(self, *args, **kwargs):
        noise = kwargs.get("noise", None)
        self.data["noise"] = self._process_with_warning(Conversions.numpy_to_pickle_bytes, noise)

    def post_epoch(self,
                   *args,
                   **kwargs
                   ) -> None:
        if not self.track_epoch:
            return
        
        noise = kwargs.get("noise", None)
        self.data["noise"] = self._process_with_warning(Conversions.numpy_to_pickle_bytes, noise)

    def serialize(self) -> Dict:
        data = {}
        
        data["noise"] = self.data["noise"]
        
        return data

    def reset_values(self) -> None:
        self.data = {
        "noise": None,
    }
        
    def _process_with_warning(self, func, *args, **kwargs):
        if args[0] is None:
            return None

        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not self.warned:
                warnings.warn(f"Error converting array to PNG: {e}")
                self.warned = True
            return None
    