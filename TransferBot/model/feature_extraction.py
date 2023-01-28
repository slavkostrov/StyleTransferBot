"""Module with descriptions of feature extraction models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .modules import Normalization, StyleLoss, ContentLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(
                channels, channels, kernel_size=3, stride=1, normalize=True, relu=True
            ),
            ConvBlock(
                channels, channels, kernel_size=3, stride=1, normalize=True, relu=False
            ),
        )

    def forward(self, x):
        return self.block(x) + x


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        upsample=False,
        normalize=True,
        relu=True,
    ):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
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


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()

        vgg_pretrained_features = models.vgg16(pretrained=True).features

        features_slices_ranges = [(0, 4), (4, 9), (9, 16), (16, 23)]

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


def get_style_model_and_losses(
    cnn,
    normalization_mean,
    normalization_std,
    style_img,
    content_img,
):
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError(
                "Unrecognized layer: {}".format(layer.__class__.__name__)
            )

        model.add_module(name, layer)

        if name in ["conv_4"]:
            # add content loss:
            target = model(content_img.to(device)).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]:
            # add style loss:
            target_feature = model(style_img.to(device)).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[: (i + 1)]

    return model, style_losses, content_losses
