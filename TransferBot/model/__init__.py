from .protocol import ModelABC, MockModel
from .vgg16 import VGG16Transfer
from .vgg19 import VGG19Transfer

MODEL_REGISTRY = {
    value.model_id: value for value in ModelABC.__subclasses__()
    if value.model_id not in ("unknown", "mock")
}
