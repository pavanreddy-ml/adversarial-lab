from typing import Dict
from . import Tracker
import numpy as np

import io
from PIL import Image



class ImageTracker(Tracker):
    columns = {
        "raw_image": "blob",
        "preprocessed_image": "blob",
        "noise": "blob",
        "noised_raw_image": "blob",
        "noised_preprocessed_image": "blob",
    }

    def __init__(self,
                 track_batch: bool = True,
                 track_epoch: bool = True,
                 track_raw_image: bool = True,
                 track_preprocessed_image: bool = True,
                 track_noise: bool = True,
                 track_noised_raw_image: bool = True,
                 track_noised_preprocessed_image: bool = True
                 ) -> None:
        super().__init__(track_batch=track_batch, track_epoch=track_epoch)
        self.track_raw_image = track_raw_image
        self.track_preprocessed_image = track_preprocessed_image
        self.track_noise = track_noise
        self.track_noised_raw_image = track_noised_raw_image
        self.track_noised_preprocessed_image = track_noised_preprocessed_image

        self.reset_values()


    def post_epoch(self,
                   epoch_num: int,
                   *args,
                   **kwargs
                   ) -> None:
        if not self.track_epoch:
            return

        raw_image = kwargs.get("raw_image", None)
        preprocessed_image = kwargs.get("preprocessed_image", None)
        noise = kwargs.get("noise", None)
        noised_raw_image = kwargs.get("noised_raw_image", None)
        noised_preprocessed_image = kwargs.get("noised_preprocessed_image", None)

        if raw_image is not None and self.track_raw_image:
            self.data["raw_image"] = self._tensor_to_png_bytes(raw_image)

        if preprocessed_image is not None and self.track_preprocessed_image:
            self.data["preprocessed_image"] = self._tensor_to_png_bytes(preprocessed_image)

        if noise is not None and self.track_noise:
            self.data["noise"] = self._tensor_to_png_bytes(noise)

        if noised_raw_image is not None and self.track_noised_raw_image:
            self.data["noised_raw_image"] = self._tensor_to_png_bytes(noised_raw_image)

        if noised_preprocessed_image is not None and self.track_noised_preprocessed_image:
            self.data["noised_preprocessed_image"] = self._tensor_to_png_bytes(noised_preprocessed_image)

    def serialize(self) -> Dict:
        data = {}
        
        if self.track_epoch:
            if self.track_raw_image:
                data["raw_image"] = self.data["raw_image"]
            
            if self.track_preprocessed_image:
                data["preprocessed_image"] = self.data["preprocessed_image"]

            if self.track_noise:
                data["noise"] = self.data["noise"]

            if self.track_noised_raw_image:
                data["noised_raw_image"] = self.data["noised_raw_image"]

            if self.track_noised_preprocessed_image:
                data["noised_preprocessed_image"] = self.data["noised_preprocessed_image"]

        return data

    def reset_values(self) -> None:
        self.data = {
            "raw_image": None,
            "preprocessed_image": None,
            "noise": None,
            "noised_raw_image": None,
            "noised_preprocessed_image": None
        }

    def _tensor_to_png_bytes(self, 
                            tensor
                            ) -> bytes:
        if tensor.shape[0] == 1 or tensor.shape[0] is None:
            numpy_array = np.array(tensor[0])
        else:
            numpy_array = np.array(tensor)

        normalized_array = ((numpy_array - numpy_array.min()) / 
                            (numpy_array.max() - numpy_array.min()) * 255).astype(np.uint8)
        
        image = Image.fromarray(normalized_array)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        return image_bytes.getvalue()