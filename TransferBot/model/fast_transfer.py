"""Module with pretrained style transfer models for fast transferring."""
import typing as tp
from abc import abstractmethod
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import transforms

from . import ModelABC
from .feature_extraction import TransformerNet

# TODO: maybe move it to class attributes
imsize = (512, 512) if torch.cuda.is_available() else (256, 256)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')


class PretrainedTransferModel(ModelABC):

    def __init__(self):
        super().__init__()
        self.max_image_size = 512
        self.model = TransformerNet()

        checkpoint_path = Path(__file__).parent.joinpath(f"checkpoints/{self.check_point_path}")

        self.transformer = TransformerNet().to(self.device)
        self.transformer.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.transformer.eval()

    def get_transforms(self) -> transforms.Compose:
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def process_image(self, content_image: BytesIO, style_image: tp.Optional[BytesIO] = None) -> BytesIO:
        image_tensor, _ = self.load_image(content_image)

        with torch.no_grad():
            stylized_image = self.denormalize(self.transformer(image_tensor)).cpu()

        return self.get_bytes_image(stylized_image)

    @staticmethod
    def denormalize(tensors):
        for c in range(3):
            tensors[:, c].mul_(std[c]).add_(mean[c])
        return tensors

    @property
    @abstractmethod
    def check_point_path(self):
        pass


class MunkModel(PretrainedTransferModel, ModelABC):
    model_id: str = "Munk"

    @property
    def check_point_path(self):
        return "munk_default.pth"


class VanGoghModel(PretrainedTransferModel, ModelABC):
    model_id: str = "Van Gogh"

    @property
    def check_point_path(self):
        return "van_gogh_default.pth"


class KandinskyModel(PretrainedTransferModel, ModelABC):
    model_id: str = "Kandinsky"

    @property
    def check_point_path(self):
        return "kandinsky_default.pth"


class MonetModel(PretrainedTransferModel, ModelABC):
    model_id = "Monet"

    @property
    def check_point_path(self):
        return "monet_default.pth"


class PicassoModel(PretrainedTransferModel, ModelABC):
    model_id: str = "Picasso"

    @property
    def check_point_path(self):
        return "picasso_default.pth"
