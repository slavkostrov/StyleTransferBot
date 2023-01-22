"""Модуль с описанием протокала для модели."""
from io import BytesIO
from typing import Protocol, runtime_checkable, Optional

from PIL import Image

from . import utils


@runtime_checkable
class ModelABC(Protocol):
    model_id: str = "unknown"

    def process_image(self, content_image: BytesIO, style_image: Optional[BytesIO] = None) -> BytesIO:
        pass

    @staticmethod
    def get_bytes_image(tensor, size):
        final_image = utils.get_pil_image(tensor)
        final_image = final_image.resize(size, Image.ANTIALIAS)
        return_image = BytesIO()
        final_image.save(return_image, "jpeg")
        return_image.seek(0)
        return return_image

    def load_image(self, filename, size, transform):
        img = Image.open(filename)
        input_image_size = img.size
        size_list = list(size)
        for i, s in enumerate(size_list):
            if s > self.max_image_size:
                size_list[i] = self.max_image_size
        size = tuple(size_list)
        image = img.resize(size, Image.ANTIALIAS)
        ssize = image.size
        image = transform(image)
        image = image.repeat(1, 1, 1, 1)
        return image, input_image_size, ssize


class MockModel(ModelABC):
    model_id: str = "mock"

    def process_image(self, img: BytesIO) -> BytesIO:
        return img
