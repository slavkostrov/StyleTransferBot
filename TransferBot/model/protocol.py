"""Module for transfer style model protocol."""
import abc
import gc
from abc import ABC
from io import BytesIO
from typing import Optional, Tuple, TypeVar

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms, Compose
from torchvision.utils import save_image

image_size = TypeVar("image_size", int, Tuple[int])


def resize_image(image: torch.tensor, size: image_size) -> torch.tensor:
    """Resize torch.tensor to given size.

    :param image: image in torch.tensor form
    :param size: size for resize
    :return: resized tensor
    """
    if isinstance(size, int):
        size: tuple = (size, size)
    x = transforms.ToPILImage()(torch.squeeze(image).cpu().clone())
    x = x.resize(size, Image.ANTIALIAS)
    x = transforms.ToTensor()(x)
    return x


class ModelABC(ABC):
    """Abstract class for all of transfer style models."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        gc.collect()

    @abc.abstractmethod
    def process_image(self, content_image: BytesIO, style_image: Optional[BytesIO] = None) -> BytesIO:
        """Process image, apply style to content or use pretrained style.

        :param content_image: image with content, will be stylized
        :param style_image: image with style which be transferred on content (optional for pretrained styles)
        :return: stylized image
        """
        pass

    @abc.abstractmethod
    def get_transforms(self) -> Compose:
        """Build transform for use after image loading.

        :return: Compose of transforms.
        """
        pass

    @abc.abstractmethod
    def model_id(self) -> str:
        pass

    @staticmethod
    def get_bytes_image(tensor: torch.tensor, size: image_size = None) -> BytesIO:
        """Save torch.tensor as jpeg image to bytes.

        :param tensor: input image torch.tensor
        :param size: size to resize, if present will be used for transform
        :return: jpeg image in BytesIO
        """
        if size is not None:
            tensor = resize_image(tensor, size=size)
        return_image = BytesIO()
        save_image(tensor, return_image, format="jpeg")
        return_image.seek(0)
        return return_image

    def load_image(self, file: BytesIO) -> Tuple[torch.tensor, image_size]:
        """Read image from bytes and apply transforms to it.

        :param file: input file with image
        :return: (image torch.tensor, size of input image)
        """
        image = Image.open(file)
        input_image_size = image.size
        transform = self.get_transforms()
        image_tensor = Variable(transform(image)).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor, input_image_size

