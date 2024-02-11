"""Model with GANs classes."""
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import Compose

from TransferBot.model import ModelABC

input_shape = (3, 512, 512)
num_residual_block = 9
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self):
        super().__init__()

        channels = input_shape[0]
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        for _ in range(2):
            out_features *= 2
            model.extend(
                [
                    nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True),
                ]
            )
            in_features = out_features

        for _ in range(num_residual_block):
            model.append(ResidualBlock(out_features))

        for _ in range(2):
            out_features //= 2
            model.extend(
                [
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
            in_features = out_features

        model.extend(
            [
                nn.ReflectionPad2d(channels),
                nn.Conv2d(out_features, channels, 7),
                nn.Tanh(),
            ]
        )

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class GANTransfer(ModelABC):
    def __init__(self):
        super().__init__()
        checkpoint_path = Path(__file__).parent.joinpath(
            f"checkpoints/{self.check_point_path}"
        )

        self.gan = GeneratorResNet().to(self.device)
        self.gan.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.gan.eval()

    @property
    @abstractmethod
    def check_point_path(self):
        pass


class MonetGAN(GANTransfer, ModelABC):
    model_id = "Monet GAN 🪄"

    @property
    def check_point_path(self):
        return "monet_GAN.pth"

    def process_image(
        self, content_image: BytesIO, style_image: Optional[BytesIO] = None
    ) -> BytesIO:
        image_tensor, size = self.load_image(content_image)
        stylized_image = self.denormalize(self.gan(image_tensor)).cpu()
        return self.get_bytes_image(stylized_image, size)

    @staticmethod
    def denormalize(tensors):
        for c in range(3):
            tensors[:, c].mul_(0.5).add_(0.5)
        return tensors

    def get_transforms(self) -> Compose:
        return transforms.Compose(
            [
                transforms.Resize(input_shape[1]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
