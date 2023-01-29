"""Module with implementations of custom torch.nn.module classes."""
import torch
import torch.nn.functional as F
from torch import nn


class ContentLoss(nn.Module):
    def __init__(self, target: torch.tensor):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x: torch.tensor) -> torch.tensor:
        self.loss = F.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    def __init__(self, target_feature: torch.tensor):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, x: torch.tensor) -> torch.tensor:
        G = self.gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x

    @staticmethod
    def gram_matrix(x: torch.tensor) -> torch.tensor:
        a, b, c, d = x.size()
        features = x.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)


class Normalization(nn.Module):
    def __init__(self, mean: torch.tensor, std: torch.tensor):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img: torch.tensor) -> torch.tensor:
        return (img - self.mean) / self.std
