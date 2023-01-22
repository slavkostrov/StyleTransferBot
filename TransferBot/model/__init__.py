from inspect import isabstract

from . import feature_extraction
from .protocol import ModelABC
from .fast_transfer import MunkModel, MonetModel, KandinskyModel, VanGoghModel
from .slow_transfer import VGG16Transfer, VGG19Transfer

MODEL_REGISTRY = {
    value.model_id: value for value in ModelABC.__subclasses__()
    if not isabstract(value) and "VGG" not in value.__name__
}
