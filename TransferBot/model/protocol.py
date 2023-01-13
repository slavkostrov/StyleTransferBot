"""Модуль с описанием протокала для модели."""
from io import BytesIO
from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelABC(Protocol):
    model_id: str = "unknown"

    def process_image(self, img: BytesIO) -> BytesIO:
        pass


class MockModel(ModelABC):
    model_id: str = "mock"

    def process_image(self, img: BytesIO) -> BytesIO:
        return img
