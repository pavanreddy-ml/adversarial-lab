import io
import pickle
import numpy as np
from PIL import Image


class Conversions:
    @staticmethod
    def numpy_to_png_bytes(array: np.ndarray,
                           normalize: bool = False
                           ) -> bytes:
        if normalize:
            array_copy = ((array - array.min()) /
                          (array.max() - array.min()) * 255).astype(np.uint8)
        else:
            array_copy = array.astype(np.uint8)

        image = Image.fromarray(array_copy)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        return image_bytes.getvalue()

    @staticmethod
    def numpy_to_pickle_bytes(array: np.ndarray) -> bytes:
        return pickle.dumps(array)
