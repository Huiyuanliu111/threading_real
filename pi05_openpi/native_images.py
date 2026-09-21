"""Lossless spatial preprocessing: 640x480 RGB -> 644x490 with black padding."""
import dataclasses

import numpy as np

SOURCE_HW = (480, 640)
MODEL_HW = (490, 644)


def pad_native_rgb(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.shape != (*SOURCE_HW, 3):
        raise ValueError(f"Expected original uint8 RGB 480x640x3, got {image.shape}/{image.dtype}")
    return np.pad(image, ((5, 5), (2, 2), (0, 0)), constant_values=0)


@dataclasses.dataclass(frozen=True)
class PadNativeImages:
    def __call__(self, data):
        result = dict(data)
        images = {}
        for key, value in data['image'].items():
            image = np.asarray(value)
            if image.shape == (*MODEL_HW, 3) and image.dtype == np.uint8:
                images[key] = image
            else:
                images[key] = pad_native_rgb(image)
        result['image'] = images
        return result
