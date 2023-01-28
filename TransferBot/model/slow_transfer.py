"""Module contains slow style transfer implementations."""
import typing as tp
from dataclasses import dataclass, field
from io import BytesIO
from logging import getLogger

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from . import ModelABC
from .feature_extraction import Vgg16, get_style_model_and_losses

LOGGER = getLogger("slow_transfer.py")

# TODO: hide all into class
imsize = (512, 512) if torch.cuda.is_available() else (256, 256)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


@dataclass
class VGG16Transfer(ModelABC):
    model_id: str = "VGG16"

    epochs: int = field(default=150)
    default_image_size: tp.Union[float, tp.Tuple[float]] = field(default=256.)

    batch_size: int = field(default=1)
    learning_rate: float = field(default=0.01)
    content_weight: float = field(default=1.)
    style_weight: float = field(default=1e9)
    tv_weight: float = field(default=2.)

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    vgg: torch.nn.Module = field(default_factory=Vgg16)

    def __post_init__(self):
        self.vgg = self.vgg.to(self.device)
        LOGGER.info(f"model is loaded. using device {self.device}.")

    def get_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(self.default_image_size),
            transforms.ToTensor(),
        ])

    def get_features(self, img):
        img = img.to(self.device)
        return self.vgg.to(self.device)(img)

    def process_image(self, content_image: BytesIO, style_image: tp.Optional[BytesIO] = None) -> BytesIO:
        if style_image is None:
            raise RuntimeError("style_image must be defined.")

        LOGGER.debug("Loading input images.")
        content_img, content_size = self.load_image(content_image)
        style_img, _ = self.load_image(style_image)
        LOGGER.debug("Getting features from model.")
        features_style, features_content = self.get_features(style_img), self.get_features(content_img)
        LOGGER.debug("Got features from model.")

        gram_style = [gram_matrix(y) for y in features_style]
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
                gram_style_y = [gram_matrix(i) for i in features_y]

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


class VGG19Transfer(ModelABC):
    model_id: str = "VGG19"

    def __init__(self, num_steps: int = 1000):
        super().__init__()
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.num_steps = num_steps

    def get_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(imsize),
            transforms.ToTensor()]
        )

    def process_image(self, content_image: BytesIO, style_image: tp.Optional[BytesIO] = None) -> BytesIO:
        if style_image is None:
            raise RuntimeError("style_image must be defined.")

        content_image, content_size = self.load_image(content_image)
        style_image, _ = self.load_image(style_image)

        output = self.run_style_transfer(
            self.cnn_normalization_mean,
            self.cnn_normalization_std,
            content_image.clone(),
            style_image,
            content_image,
            num_steps=self.num_steps,
        )
        return self.get_bytes_image(output, content_size)

    @staticmethod
    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def run_style_transfer(
            self,
            normalization_mean,
            normalization_std,
            content_img,
            style_img,
            input_img,
            num_steps=5000,
            style_weight=10000000,
            content_weight=0.4
    ):
        """Run the style transfer."""
        LOGGER.info('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(
            self.cnn,
            normalization_mean,
            normalization_std,
            style_img,
            content_img
        )

        input_img.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = self.get_input_optimizer(input_img)

        LOGGER.info('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img.to(self.device))
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    LOGGER.info("run {}:".format(run))
                    LOGGER.info(
                        'Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))

                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img
