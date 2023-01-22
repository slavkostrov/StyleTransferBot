import typing as tp
from io import BytesIO
from logging import getLogger

import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from . import ModelABC
from .utils import Normalization, StyleLoss, ContentLoss

LOGGER = getLogger("vgg19.py")

# TODO: hide all into class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = (512, 512) if torch.cuda.is_available() else (256, 256)  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()]
)  # transform it into a torch tensor

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(
        cnn,
        normalization_mean,
        normalization_std,
        style_img,
        content_img,
        content_layers=content_layers_default,
        style_layers=style_layers_default
):
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img.to(device)).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img.to(device)).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=100000, content_weight=1):
    """Run the style transfer."""
    LOGGER.info('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img,
                                                                     content_img)

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    LOGGER.info('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img.to(device))
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
                LOGGER.info('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


class VGG19Transfer(ModelABC):
    model_id: str = "VGG19"

    def __init__(self):
        super().__init__()
        self.max_image_size = 512
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()

    def process_image(self, content_image: BytesIO, style_image: tp.Optional[BytesIO] = None) -> BytesIO:
        if style_image is None:
            response = requests.get(
                "https://uploads4.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!Large.jpg")
            style_image = BytesIO(response.content)

        content_image, content_size, result_size = self.load_image(content_image, imsize, loader)
        style_image, _, _ = self.load_image(style_image, imsize, loader)

        output = run_style_transfer(
            self.cnn,
            cnn_normalization_mean,
            cnn_normalization_std,
            content_image.clone(),
            style_image,
            content_image
        )
        return self.get_bytes_image(output, result_size)
