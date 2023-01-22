from unittest.mock import AsyncMock

import pytest

from TransferBot.bot import TransferBot
from TransferBot.bot import bot_answers

# bot = TransferBot(bot_token="TOKEN")


@pytest.mark.asyncio
async def test_echo_handler():
    chat_mock = AsyncMock(id="001", first_name="name")
    message_mock = AsyncMock(text="/start", chat=chat_mock)

    await TransferBot.send_welcome(message=message_mock)
    message_mock.reply.assert_called_with(bot_answers.welcome_message.format(message=message_mock), parse_mode='MarkdownV2')


