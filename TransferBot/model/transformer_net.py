import typing as tp
from abc import abstractmethod
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision.utils import save_image

from TransferBot.model import ModelABC

# TODO: maybe move it to class attributes
imsize = (512, 512) if torch.cuda.is_available() else (256, 256)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def get_test_transform(image_size=None):
    """ Transforms for test image """
    resize = [transforms.Resize(image_size)] if image_size else []
    transform = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transform


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=True),
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=False),
        )

    def forward(self, x):
        return self.block(x) + x


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2), nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if normalize else None
        self.relu = relu

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        x = self.block(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x)
        return x


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, stride=1, normalize=False, relu=False),
        )

    def forward(self, x):
        return self.model(x)


class PretrainedTransferModel(ModelABC):

    def __init__(self):
        super().__init__()
        self.max_image_size = 512
        self.model = TransformerNet()

        checkpoint_path = Path(__file__).parent.joinpath(f"checkpoints/{self.check_point_path}")
        self.transform = get_test_transform()

        self.transformer = TransformerNet().to(self.device)
        self.transformer.load_state_dict(torch.load(checkpoint_path))
        self.transformer.eval()

    def process_image(self, content_image: BytesIO, style_image: tp.Optional[BytesIO] = None) -> BytesIO:
        image_tensor = Variable(self.transform(Image.open(content_image))).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            stylized_image = self.denormalize(self.transformer(image_tensor)).cpu()

        return_image = BytesIO()
        save_image(stylized_image, return_image, format="jpeg")
        return_image.seek(0)
        return return_image

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
        return "munk_last_checkpoint.pth"


class VanGoghModel(PretrainedTransferModel, ModelABC):
    model_id: str = "Van Gogh"

    @property
    def check_point_path(self):
        return "van_gog_last_checkpoint.pth"


class KandinskyModel(PretrainedTransferModel, ModelABC):
    model_id: str = "Kandinsky"

    @property
    def check_point_path(self):
        return "kandinsky_last_checkpoint.pth"


class MonetModel(PretrainedTransferModel, ModelABC):
    model_id = "Monet"

    @property
    def check_point_path(self):
        return "mone2_4800.pth"


class PicassoModel(PretrainedTransferModel, ModelABC):
    model_id: str = "Picasso"

    @property
    def check_point_path(self):
        return "best_model.pth"
