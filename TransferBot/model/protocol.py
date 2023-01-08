"""Модуль с описанием протокала для модели."""
from io import BytesIO
from typing import Protocol


class ModelABC(Protocol):
    async def process_image(self, img: BytesIO) -> BytesIO:
        pass


class MockModel(ModelABC):
    async def process_image(self, img: BytesIO) -> BytesIO:
        return img
