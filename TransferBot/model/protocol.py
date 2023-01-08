"""Модуль с описанием протокала для модели."""
from io import BytesIO
from typing import Protocol


class ModelABC(Protocol):
    def process_image(self, img: BytesIO) -> BytesIO:
        pass


class MockModel(ModelABC):
    def process_image(self, img: BytesIO) -> BytesIO:
        return img
