from inspect import isabstract

from .protocol import ModelABC
from .vgg16 import VGG16Transfer
from .vgg19 import VGG19Transfer
from .transformer_net import MunkModel, MonetModel, KandinskyModel, VanGoghModel


MODEL_REGISTRY = {
    value.model_id: value for value in ModelABC.__subclasses__()
    if not isabstract(value)
}
