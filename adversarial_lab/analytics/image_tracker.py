from typing import Dict
from . import Tracker

import warnings

from adversarial_lab.utils import Conversions


class ImageTracker(Tracker):
    _columns = {
        "original_image": "blob",
        "preprocessed_image": "blob",
    }

    def __init__(self,
                 ) -> None:
        super().__init__()
        self.warned = False

    def pre_attack(self, *args, **kwargs):
        original_image = kwargs.get("original_sample", None)
        preprocessed_image = kwargs.get("preprocessed_sample", None)

        self.data["original_image"] = self._process_with_warning(Conversions.numpy_to_png_bytes, original_image)
        self.data["preprocessed_image"] = self._process_with_warning(Conversions.numpy_to_pickle_bytes, preprocessed_image)

    def post_epoch(self,
                   *args,
                   **kwargs
                   ) -> None:
        pass

    def serialize(self) -> Dict:
        data = {}
        
        data["original_image"] = self.data["original_image"]
        data["preprocessed_image"] = self.data["preprocessed_image"]

        return data

    def reset_values(self) -> None:
        self.data = {
        "original_image": None,
        "preprocessed_image": None,
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
    