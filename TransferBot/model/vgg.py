import sys
import typing as tp
from dataclasses import dataclass, field
from io import BytesIO
from logging import getLogger

import requests
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models
from torchvision import transforms

sys.path.append("C:\\Users\\Xiaomi\\Documents\\dls_project_2022_23\\TransferBot\\model\\")
import utils
from TransferBot.model import ModelABC

LOGGER = getLogger(__file__)


class Vgg16(torch.nn.Module):

    def __init__(self):
        super(Vgg16, self).__init__()

        vgg_pretrained_features = models.vgg16(pretrained=True).features

        features_slices_ranges = [
            (0, 4),
            (4, 9),
            (9, 16),
            (16, 13)
        ]

        self.features_slices = []
        for slice_id, (start, end) in enumerate(features_slices_ranges):
            feature_slice = torch.nn.Sequential()
            setattr(self, f"slice_{slice_id}", feature_slice)
            for i in range(start, end):
                vgg_slice = vgg_pretrained_features[i]
                vgg_slice.requires_grad = False
                feature_slice.add_module(f"slice_{slice_id}_{i}", vgg_slice)
            self.features_slices.append(feature_slice)

    def forward(self, X):
        out = list()
        for feature_slice in self.features_slices:
            X = feature_slice(X)
            out.append(X)
        return out


@dataclass
class VGGTransfer(ModelABC):
    model_id: str = "VGG16"

    epochs: int = field(default=150)
    default_image_size: int = field(default=256)
    max_image_size: int = field(default=512)

    batch_size: int = field(default=1)
    learning_rate: float = field(default=0.01)
    content_weight: float = field(default=1.)
    style_weight: float = field(default=1e9)
    tv_weight: float = field(default=2.)

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    vgg: torch.nn.Module = field(default_factory=lambda: Vgg16())

    @property
    def image_size(self):
        size = self.default_image_size
        if isinstance(size, int):
            size = (size, size)
        return size

    def __post_init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

        self.style_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

        self.vgg = self.vgg.to(self.device)
        LOGGER.info("model is loaded.") # LOGGER.info(f"Model: {self}")

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

    def get_features(self, img):
        img = img.to(self.device)
        return Vgg16().to(self.device)(img)

    @staticmethod
    def get_bytes_image(tensor, size):
        final_image = utils.get_pil_image(tensor)
        final_image = final_image.resize(size, Image.ANTIALIAS)
        return_image = BytesIO()
        final_image.save(return_image, "jpeg")
        return_image.seek(0)
        return return_image

    def process_image(self, content_image: BytesIO, style_image: tp.Optional[BytesIO] = None) -> BytesIO:
        if style_image is None:
            response = requests.get(
                "https://uploads4.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!Large.jpg")
            style_image = BytesIO(response.content)

        content_img, content_size, result_size = self.load_image(content_image, self.image_size, self.transforms)
        style_img, _, _ = self.load_image(style_image, result_size, self.style_transform)
        features_style, features_content = self.get_features(style_img), self.get_features(content_img)

        gram_style = [utils.gram_matrix(y) for y in features_style]
        mse_loss = nn.MSELoss()

        y = content_img.detach()
        y = y.requires_grad_()
        optimizer = optim.Adam([y], lr=self.learning_rate)

        LOGGER.info("Style transfer is started.")
        for epoch in torch.arange(self.epochs):
            def closure():
                optimizer.zero_grad()
                y.data.clamp_(0, 1)
                features_y = self.vgg(y.to(self.device))
                gram_style_y = [utils.gram_matrix(i) for i in features_y]

                fc = features_content[-1]
                fy = features_y[-1]

                style_loss = 0
                for fy_gm, fs_gm in zip(gram_style_y, gram_style):
                    style_loss += mse_loss(fy_gm, fs_gm)
                style_loss = self.style_weight * style_loss

                content_loss = self.content_weight * mse_loss(fc, fy)

                tv_loss = self.tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                            torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

                total_loss = content_loss + style_loss + tv_loss
                total_loss.backward(retain_graph=True)

                if epoch % 100 == 0:
                    LOGGER.info("Epoch {}: Style Loss : {:4f} Content Loss: {:4f} TV Loss: {:4f}".format(
                        epoch,
                        style_loss,
                        content_loss,
                        tv_loss,
                        )
                    )

                return total_loss

            optimizer.step(closure)

        return self.get_bytes_image(y, content_size)
