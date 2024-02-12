# import sys
# sys.path.append("/home/slava/hse-mlds/subjects/python-advanced/homeworks/tests/")
from typing import Iterator
import random
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from aiogram.types import CallbackQuery, Chat, MediaGroup, Message, PhotoSize, User
from PIL import Image

from transferbot.bot.transfer_bot import TransferBot
from transferbot.model import MODEL_REGISTRY

DEFAULT_STYLE_REPLY_MESSAGE_ID = 123
DEFAULT_MODELS_NUMBER = 3
DEFAULT_USER_ID = 123

pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="session", autouse=True)
def default_session_fixture() -> Iterator[None]:
    """Создаёт глобальный Mock внешних зависимостей aiogram.
    
    Создаёт Mock-объекты для:
        * aiogram.Bot
        * aiogram.Dispatcher
        * aiogram.executor
        
    Дополнительно добавляется Mock для скачивания фото - записывается генерируемое 
    с помощью PIL изображение. В дочерних Mock объектах выставляется message_id.
    
    Таким образом, достигается возможноть тестировния объекта TransferBot без использования
    вннешниъ aiogram зависимостей, требующих подключение к интернету, токен и так далее.
    """
    with (
        patch("transferbot.bot.transfer_bot.Bot") as bot_mock,
        patch("transferbot.bot.transfer_bot.Dispatcher") as dispatcher_mock,
        patch("transferbot.bot.transfer_bot.executor") as executor_mock,
    ):
        bot_mock.return_value = AsyncMock()
        
        def download_image_mock(arg, *args, **kwargs):
            image = Image.new("RGB", (300, 50))
            image.save(arg, 'PNG')
        
        bot_mock.return_value.get_file.return_value = AsyncMock()
        bot_mock.return_value.edit_message_text.return_value = AsyncMock()
        bot_mock.return_value.get_file.return_value.download.side_effect = download_image_mock
        
        # TODO: можно ставить рандомное и внутри теста вытягивать какое получилось
        # для проверки очередей фигачить в цикле измнение этого значения
        bot_mock.return_value.edit_message_text.return_value.message_id = DEFAULT_STYLE_REPLY_MESSAGE_ID

        yield (
            bot_mock, 
            dispatcher_mock,
            executor_mock,
        )
     
def get_message_mock(**extras) -> Message:
    """Создаёт объект типа aiogram.types.Message и Mock'ает метод reply."""
    default_kwargs = dict(
        message_id=random.randint(1, 1_000_000),
        chat=Chat(id=DEFAULT_USER_ID),
    )
    kwargs = {
        **default_kwargs,
        **extras,
    }
    message = Message(**kwargs)
    message.reply = AsyncMock()
    return message


def get_callback_query_mock(model_id: str, user_id: int, message_id: int):
    """Создаёт объект типа aiogram.types.CallbackQuery."""
    query = CallbackQuery(
        message=Message(message_id=message_id, user_id=user_id),
        data=f":{model_id}:{message_id}",
    )
    query.from_user = User(id=user_id)
    return query


@pytest_asyncio.fixture()
async def bot():
    """Фикстура, которая создаёт объект TransferBot, при старте вызывает метод on_startup,
    а при выключении вызывает метод on_shutdown.
    """
    bot = TransferBot(bot_token="TEST_TOKEN", slow_transfer_iters=10)
    await bot.on_startup()
    bot.run()
    yield bot
    await bot.on_shutdown()


# TODO: use FiltersFactory
async def get_relevant_message_handler(bot_instance, message: Message, message_handler: bool = True) -> callable:
    dispatcher = bot_instance.dispatcher
    if message_handler:
        handlers = dispatcher.register_message_handler
    else:
        handlers = dispatcher.register_callback_query_handler
    for handler in handlers.call_args_list:
        args = handler.args
        
        func = args[0]
        filter_instance = None
        if len(args) > 1:
            filter_instance = args[1]
        
        expected_commands = handler.kwargs.get("commands")
        expected_content_types = handler.kwargs.get("content_types")

        conditions = [True]
        if filter_instance:
            conditions.append(await filter_instance.check(message))
        
        if expected_commands:
            conditions.append(message.text in expected_commands)

        if expected_content_types:
            conditions.append(message.content_type in expected_content_types)
            
        if all(conditions):
            return func
    

async def test_bot_create_aiogram_instances_correctly(bot):
    bot.bot._mock_new_parent.assert_called_with(token="TEST_TOKEN"), (
        "internal aiogram not created with token."
    )

    bot.dispatcher._mock_new_parent.assert_called_with(bot.bot), (
        "internal aiogram dispatcher not created with bot instance."
    )

    bot.dispatcher.register_message_handler.assert_called(), (
        "there is no setup message handlers calls inside bot init."
    )

    bot.dispatcher.register_callback_query_handler.assert_called(), (
        "there is no setup callback handlers calls inside bot init."
    )

async def test_bot_can_answer_on_start_command(bot):
    message_mock = get_message_mock(text="start")
    
    expected_handler = TransferBot.send_welcome
    actual_handler = await get_relevant_message_handler(bot, message_mock)
    assert expected_handler is actual_handler, (
        f"expected welcome handler is different, got {actual_handler}."
    )
    await actual_handler(message_mock)

    message_mock.reply.assert_called_once()
    assert "Hi" in message_mock.reply.call_args_list[0].args[0], (
        "`Hi` not found in welcome message."
    )

async def test_bot_can_return_help_message_with_models_info(bot):
    message_mock = get_message_mock(text="help")
    
    expected_handler = TransferBot.help_handler
    actual_handler = await get_relevant_message_handler(bot, message_mock)
    assert expected_handler is actual_handler, (
        f"expected help handler is different, got {actual_handler}."
    )
    
    await actual_handler(message_mock)
    message_mock.reply.assert_called_once()
    assert "you need to send me any picture" in message_mock.reply.call_args_list[0].args[0], (
        "help message not found in reply of help handler."
    )
    reply_message_mock = message_mock.reply.return_value
    reply_message_mock.reply_media_group.assert_called()
    
    media_group: MediaGroup = reply_message_mock.reply_media_group.call_args_list[0]
    actual_models_count = len(media_group.media)
    expected_models_count = DEFAULT_MODELS_NUMBER

    assert actual_models_count == expected_models_count, (
        "count of examples isn't equal to real models number."
    )


async def test_bot_can_handle_unknown_messages(bot):
    message_mock = get_message_mock(text="UNKNOWN_TEXT")
    
    expected_handler = TransferBot.unknown_handler
    actual_handler = await get_relevant_message_handler(bot, message_mock)
    assert expected_handler is actual_handler, (
        f"expected unknown handler is different, got {actual_handler}."
    )
    
    await actual_handler(message_mock)
    message_mock.reply.assert_called_once()
    # TODO: use called with?
    assert "end an image or use the commands" in message_mock.reply.call_args_list[0].args[0], (
        "incorrect answer to unknown message."
    )

@pytest.mark.parametrize("model_id", [*MODEL_REGISTRY.keys(), "Your style"])
async def test_bot_can_process_content_photo_correctly(bot, model_id):
    message_mock = get_message_mock(photo=[PhotoSize(file_id=DEFAULT_USER_ID)])
    bot.bot.return_value.edit_message_text.return_value.message_id = message_mock.message_id
    
    expected_handler = TransferBot.process_content_photo
    actual_handler = await get_relevant_message_handler(bot, message_mock)
    assert expected_handler.__name__ is actual_handler.__name__, (
        f"expected content photo handler is different, got {actual_handler}."
    )
    
    await actual_handler(message_mock)
    
    callback_query_mock = get_callback_query_mock(
        model_id=model_id,
        user_id=DEFAULT_USER_ID,
        message_id=message_mock.message_id,
    )

    await bot.process_model_selection(callback_query_mock)
    
    # TODO:
    # Add asserts
    if model_id == "Your style":
        message_mock.reply_to_message = get_message_mock(message_id=DEFAULT_STYLE_REPLY_MESSAGE_ID)
        expected_handler = TransferBot.process_style_photo
        actual_handler = await get_relevant_message_handler(bot, message_mock)
        assert expected_handler.__name__ is actual_handler.__name__, (
            f"expected style photo handler is different, got {actual_handler}."
        )
        await actual_handler(message_mock)
        
        # тест очередей
        # import asyncio
        # await asyncio.gather(*[actual_handler(message_mock) for i in range(10)])


# TODO: message с разными рандомными id
# TODO: добавить ассерты
# TODO: разделить your style и не your style в разные тесты
# TODO: добавить проверки очереди (проверять кол-во дочерних процессов?)
# TODO: отформотировать код
# TODO: подумать над резолвом фильтров
# TODO: добавить докстринги
