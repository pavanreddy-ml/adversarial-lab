from typing import Dict
from . import Tracker
import numpy as np
import pickle

import io
from PIL import Image

import warnings


class ImageTracker(Tracker):
    columns = {
        "raw_image": "blob",
        "preprocessed_image": "blob",
        "true_noise_raw_image": "blob",
        "true_noise_preprocessed_image": "blob",
        "normalized_noise_raw_image": "blob",
        "normalized_noise_preprocessed_image": "blob",
        "noised_raw_image": "blob",
        "noised_preprocessed_image": "blob"
    }

    def __init__(self,
                 track_raw_image: bool = True,
                 track_preprocessed_image: bool = True,
                 track_true_noise_raw_image: bool = True,
                 track_true_noise_preprocessed_image: bool = True,
                 track_normalized_noise_raw_image: bool = True,
                 track_normalized_noise_preprocessed_image: bool = True,
                 track_nooised_raw_image: bool = True,
                 track_noised_preprocessed_image: bool = True,
                 ) -> None:
        super().__init__()
        self.track_raw_image = track_raw_image
        self.track_preprocessed_image = track_preprocessed_image
        self.track_true_noise_raw_image = track_true_noise_raw_image
        self.track_true_noise_preprocessed_image = track_true_noise_preprocessed_image
        self.track_normalized_noise_raw_image = track_normalized_noise_raw_image
        self.track_normalized_noise_preprocessed_image = track_normalized_noise_preprocessed_image
        self.track_noised_raw_image = track_nooised_raw_image
        self.track_noised_preprocessed_image = track_noised_preprocessed_image

        self.warned = False


    def post_epoch(self,
                   epoch_num: int,
                   *args,
                   **kwargs
                   ) -> None:
        if not self.track_epoch:
            return

        raw_image = kwargs.get("raw_image", None)
        preprocessed_image = kwargs.get("preprocessed_image", None)
        noise_raw_image = kwargs.get("noise_raw", None)
        noise_preprocessed_image = kwargs.get("noise_preprocessed_image", None)

        if self.track_raw_image:
            self.data["raw_image"] = self._numpy_to_png_bytes(raw_image)

        if self.track_preprocessed_image:
            self.data["preprocessed_image"] = self._numpy_to_pickle_bytes(preprocessed_image)

        if self.track_true_noise_raw_image:
            self.data["true_noise_raw_image"] = self._numpy_to_pickle_bytes(noise_raw_image)

        if self.track_true_noise_preprocessed_image:
            self.data["true_noise_preprocessed_image"] = self._numpy_to_pickle_bytes(noise_preprocessed_image)

        if self.track_normalized_noise_raw_image:
            self.data["normalized_noise_raw_image"] = self._numpy_to_png_bytes(noise_raw_image, normalize=True)

        if self.track_normalized_noise_preprocessed_image:
            self.data["normalized_noise_preprocessed_image"] = self._numpy_to_png_bytes(noise_preprocessed_image, normalize=True)

        if self.track_noised_raw_image:
            if raw_image is None or noise_raw_image is None:
                self.data["noised_raw_image"] = None
            else:
                self.data["noised_raw_image"] = self._numpy_to_png_bytes(raw_image + noise_raw_image)

        if self.track_noised_preprocessed_image:
            if preprocessed_image is None or noise_preprocessed_image is None:
                self.data["noised_preprocessed_image"] = None
            else:
                self.data["noised_preprocessed_image"] = self._numpy_to_pickle_bytes(preprocessed_image + noise_preprocessed_image)

        if not self.warned:
            self.warned = True

    def serialize(self) -> Dict:
        data = {}
        
        if self.track_raw_image:
            data["raw_image"] = self.data["raw_image"]
        
        if self.track_preprocessed_image:
            data["preprocessed_image"] = self.data["preprocessed_image"]

        if self.track_true_noise_raw_image:
            data["true_noise_raw_image"] = self.data["true_noise_raw_image"]
        
        if self.track_true_noise_preprocessed_image:
            data["true_noise_preprocessed_image"] = self.data["true_noise_preprocessed_image"]

        if self.track_normalized_noise_raw_image:
            data["normalized_noise_raw_image"] = self.data["normalized_noise_raw_image"]

        if self.track_normalized_noise_preprocessed_image:
            data["normalized_noise_preprocessed_image"] = self.data["normalized_noise_preprocessed_image"]

        if self.track_noised_raw_image:
            data["noised_raw_image"] = self.data["noised_raw_image"]

        if self.track_noised_preprocessed_image:
            data["noised_preprocessed_image"] = self.data["noised_preprocessed_image"]

        return data

    def reset_values(self) -> None:
        self.data = {
        "raw_image": None,
        "preprocessed_image": None,
        "true_noise_raw_image": None,
        "true_noise_preprocessed_image": None,
        "normalized_noise_raw_image": None,
        "normalized_noise_preprocessed_image": None,
        "noised_raw_image": None,
        "noised_preprocessed_image": None,
    }
        
    def _remove_batch_dimension(self,
                                array: np.ndarray
                                ) -> np.ndarray:
        if array.shape[0] == 1 or array.shape[0] is None:
            return np.array(array[0])
        return np.array(array)

    def _numpy_to_png_bytes(self, 
                            array: np.ndarray,
                            normalize: bool = False
                            ) -> bytes:
        if array is None:
            return None
        
        try:
            array_copy = self._remove_batch_dimension(array)

            if normalize:
                array_copy = ((array_copy - array_copy.min()) / 
                            (array_copy.max() - array_copy.min()) * 255).astype(np.uint8)

            image = Image.fromarray(array_copy)
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            return image_bytes.getvalue()
        except Exception as e:
            if not self.warned:
                warnings.warn(f"Error converting array to PNG: {e}")
                self.warned = True
            return None
        
    def _numpy_to_pickle_bytes(self, 
                               array: np.ndarray
                               ) -> bytes:
        if array is None:
            return None
        
        try:
            array = self._remove_batch_dimension(array)
            return pickle.dumps(array)
        except Exception as e:
            if not self.warned:
                warnings.warn(f"Error converting array to pickle: {e}")
            return None
        
