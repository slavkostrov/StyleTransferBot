from .protocol import ModelABC, MockModel
from .vgg import VGGTransfer
from .vgg2 import VGG19Transfer

MODEL_REGISTRY = {
    value.__name__: value for value in ModelABC.__subclasses__()
}
