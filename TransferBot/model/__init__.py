from .protocol import ModelABC, MockModel
from .vgg import VGGTransfer
from .vgg2 import VGG19Transfer

MODEL_REGISTRY = {
    value.model_id: value for value in ModelABC.__subclasses__()
    if value.model_id not in ("unknown", "mock")
}
