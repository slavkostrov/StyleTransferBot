import typing as tp
from abc import abstractmethod
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision.utils import save_image

from TransferBot.model import ModelABC

# TODO: maybe move it to class attributes
from .feature_extraction import TransformerNet

imsize = (512, 512) if torch.cuda.is_available() else (256, 256)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def get_test_transform(image_size=None):
    """ Transforms for test image """
    resize = [transforms.Resize(image_size)] if image_size else []
    transform = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transform


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
