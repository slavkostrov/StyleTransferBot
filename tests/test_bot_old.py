from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import torch
from PIL import Image

from transferbot.bot import TransferBot, bot_answers
from transferbot.model import MODEL_REGISTRY, VGG19Transfer
from transferbot.model.protocol import resize_image

test_data_path = Path(__file__).parent / "test_data"


@pytest.mark.asyncio
async def test_echo_handler():
    chat_mock = AsyncMock(id="001", first_name="name")
    message_mock = AsyncMock(text="/start", chat=chat_mock)

    await TransferBot.send_welcome(message=message_mock)
    message_mock.reply.assert_called_with(
        bot_answers.welcome_message.format(message=message_mock),
        parse_mode="MarkdownV2",
    )


def test_image_resize():
    input_image = torch.rand(3, 64, 128)
    for sizes in [(64, 64), (32, 64), (64, 32)]:
        resized_image = resize_image(input_image, tuple(sizes[::-1]))
        assert tuple(resized_image.shape) == (3, *sizes)


@pytest.mark.parametrize("model_class", list(MODEL_REGISTRY.values()))
def test_pretrained_models_simple(model_class):
    test_image_path = test_data_path / "1.jpg"
    with open(test_image_path, "rb") as fh:
        test_image = BytesIO(fh.read())

    model = model_class()
    result_image = model.process_image(test_image)
    assert isinstance(result_image, BytesIO)


@pytest.mark.parametrize("model_class", list(MODEL_REGISTRY.values()))
def test_pretrained_models_sizes(model_class):
    test_image_path = test_data_path / "1.jpg"
    with open(test_image_path, "rb") as fh:
        test_image = BytesIO(fh.read())

    input_size = Image.open(test_image).size
    model = model_class()
    result_image = model.process_image(test_image)
    result_size = Image.open(result_image).size

    assert input_size[0] == pytest.approx(result_size[0], 10)
    assert input_size[1] == pytest.approx(result_size[1], 10)


@pytest.mark.slowtest
def test_slow_transfer():
    content_image_path = test_data_path / "1.jpg"
    style_image_path = test_data_path / "2.jpg"

    with open(content_image_path, "rb") as fh:
        content_image = BytesIO(fh.read())

    with open(style_image_path, "rb") as fh:
        style_image = BytesIO(fh.read())

    model = VGG19Transfer(num_steps=50)
    result_image = model.process_image(content_image, style_image)

    assert isinstance(result_image, BytesIO)
