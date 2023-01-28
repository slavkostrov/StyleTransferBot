from unittest.mock import AsyncMock

import pytest
import torch

from TransferBot.bot import TransferBot
from TransferBot.bot import bot_answers
from TransferBot.model.protocol import resize_image


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
