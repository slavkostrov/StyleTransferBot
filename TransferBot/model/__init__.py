"""Package with models desctiptions."""
import typing as tp
from inspect import isabstract

from . import feature_extraction
from .protocol import ModelABC
from .fast_transfer import KandinskyModel, MonetModel, MunkModel, VanGoghModel
from .gan_transfer import MonetGAN
from .slow_transfer import VGG16Transfer, VGG19Transfer

MODEL_REGISTRY: tp.Dict[str, tp.Type[ModelABC]] = {
    value.model_id: value
    for value in ModelABC.__subclasses__()
    if not isabstract(value) and "VGG" not in value.__name__
}


def register_model(model_class: type, model_id: str) -> tp.NoReturn:
    """Register custom model to use it in bot.

    :param model_class: custom model class.
    :param model_id: model_id, string name for model.
    :return: None
    """
    if not all(
        hasattr(model_class, name) for name in dir(ModelABC) if not name.startswith("_")
    ) or isabstract(model_class):
        raise TypeError("model_class must be inherited from ModelABC.")
    MODEL_REGISTRY[model_id] = model_class
