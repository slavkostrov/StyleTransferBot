import asyncio
import sys

sys.path.append("/home/slava/hse-mlds/subjects/python-advanced/homeworks/tests/")
import random
import typing as tp
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
import pytest_asyncio
from aiogram.types import CallbackQuery, Chat, MediaGroup, Message, PhotoSize, User
from PIL import Image

from transferbot.bot.bot_answers import unknown_message, welcome_message
from transferbot.bot.transfer_bot import TransferBot
from transferbot.model import MODEL_REGISTRY

DEFAULT_MODEL_ID_LIST = tuple(MODEL_REGISTRY.keys())
DEFAULT_YOUR_STYLE_MODEL_ID = "Your style"
DEFAULT_MAX_MESSAGES_NUMBER = 100
DEFAULT_TEST_TOKEN = "TEST_TOKEN"
DEFAULT_MODELS_NUMBER = 3
DEFAULT_USER_ID = 123
DEFAULT_NUMBER_OF_PARALLEL_REQUESTS = 10

pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="function", autouse=True)
def default_session_fixture() -> tp.Generator[tp.Any, tp.Any, tp.Any]:
    """Создаёт глобальный Mock внешних зависимостей aiogram.
    
    Создаёт Mock-объекты для:
        * aiogram.Bot
        * aiogram.Dispatcher
        * aiogram.executor
        
    Дополнительно добавляется Mock для скачивания фото - записывается генерируемое 
    с помощью PIL изображение. В дочерних Mock объектах выставляется message_id, который
    для удобства тестирования будет равен порядковому номеру сообщения (подробнее ниже).
    
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
        
        # эмулируем различные message_id для ответов, это необходимо для того, чтобы
        # протестировавать параллельную обработку нескольких запросов на перенос стиля
        # таким образом, каждый раз будет возвращаться число (message_id) на 1 большее предыдущего
        type(bot_mock.return_value.edit_message_text.return_value).message_id = PropertyMock(
            side_effect=range(DEFAULT_MAX_MESSAGES_NUMBER),
        )

        yield (
            bot_mock, 
            dispatcher_mock,
            executor_mock,
        )
     
def get_message_mock(**extras) -> Message:
    """Создаёт объект типа aiogram.types.Message и Mock'ает метод reply."""
    default_kwargs = dict(
        message_id=random.getrandbits(128),
        chat=Chat(id=DEFAULT_USER_ID),
    )
    kwargs = {
        **default_kwargs,
        **extras,
    }
    message = Message(**kwargs)
    message.reply = AsyncMock()
    return message


def get_callback_query_mock(model_id: str, user_id: int, message_id: int) -> CallbackQuery:
    """Создаёт объект типа aiogram.types.CallbackQuery."""
    query = CallbackQuery(
        message=Message(message_id=message_id, user_id=user_id),
        data=f":{model_id}:{message_id}",
    )
    query.from_user = User(id=user_id)
    return query


@pytest_asyncio.fixture()
async def bot() -> tp.AsyncGenerator[TransferBot, tp.Any]:
    """Фикстура, которая создаёт объект TransferBot, при старте вызывает метод on_startup,
    а при выключении вызывает метод on_shutdown.
    """
    bot = TransferBot(bot_token=DEFAULT_TEST_TOKEN, slow_transfer_iters=10)
    await bot.on_startup()
    bot.run()
    yield bot
    await bot.on_shutdown()



async def get_relevant_handler(bot_instance: TransferBot, message: Message, message_handler: bool = True) -> tp.Callable:
    """Реализует подбор хэндлера в зависимости от сообщения.
    
    По своей сути повторяет логику, которую реализует aiogram при получении нового сообщения, подбирая
    подходящий оброботчик. Для юнит-тестирования мы хотим тестировать именно наши хэндлеры, поэтому 
    подбор хэндлеров пишется явно и не тестируется.
    """
    dispatcher = bot_instance.dispatcher

    if message_handler:
        # если хотим найти хэндлер для сообщения
        handlers = dispatcher.register_message_handler
    else:
        # если хотим найти хэндлер для callback_query
        handlers = dispatcher.register_callback_query_handler

    # проходимся по всем хэндлерам и подбираем подходящий
    for handler in handlers.call_args_list:
        args = handler.args
        
        func = args[0]
        filter_instance = None
        if len(args) > 1:
            # извлекаем фильтр из аргументов
            filter_instance = args[1]
        
        # достаём команду и тип контента, которые были указаны для хэндлера
        expected_commands = handler.kwargs.get("commands")
        expected_content_types = handler.kwargs.get("content_types")

        conditions = [True]
        if filter_instance:
            # проверяем фильтр
            conditions.append(await filter_instance.check(message))
        
        if expected_commands:
            # провряем команду
            conditions.append(message.text in expected_commands)

        if expected_content_types:
            # проверяем тип контента
            conditions.append(message.content_type in expected_content_types)
        
        # если все условия для сообщения пройдены, то возвращаем хэндлер
        if all(conditions):
            return func


async def check_correct_handler(
    bot: TransferBot,
    obj: tp.Union[Message, CallbackQuery],
    expected_handler: tp.Callable,
) -> tp.Callable:
    """Проверяет корректность выбора обработчика."""
    is_message = isinstance(obj, Message)
    actual_handler = await get_relevant_handler(bot, obj, message_handler=is_message)
    assert expected_handler is getattr(TransferBot, actual_handler.__name__), (
        f"expected handler is different, expect {expected_handler}, but got {actual_handler}."
    )
    return actual_handler

def assert_correct_result_photo(message: Message, kwargs: tp.Dict[str, tp.Any], model_id: str):
    assert DEFAULT_USER_ID == kwargs.get("chat_id"), "wrong reply chat_id"
    assert "photo" in kwargs.keys(), "there is no photo in reply"
    assert "caption" in kwargs.keys(), "trere is no caption in reply"
    assert model_id in kwargs["caption"], "there is no model_id in caption text"
    assert message.message_id == kwargs["reply_to_message_id"], "message isn't reply to request message"


async def test_bot_create_aiogram_instances_correctly(bot: TransferBot):
    """Проверяем, что внутри класса TransferBot создаются все нужные aiogram сущности."""
    bot.bot._mock_new_parent.assert_called_with(token=DEFAULT_TEST_TOKEN), (
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

async def test_bot_can_answer_on_start_command(bot: TransferBot):
    """Проверяем корректность приветственного сообщения."""
    message_mock = get_message_mock(text="start")
    
    # 1. проверяем, что используется нужный хэндлер и вызываем его
    actual_handler = await check_correct_handler(bot, message_mock, TransferBot.send_welcome)
    await actual_handler(message_mock)

    # 2. проверяем, что отправлено сообщение с правильным текстом
    message_mock.reply.assert_called_once_with(welcome_message.format(message=message_mock), parse_mode="MarkdownV2")


async def test_bot_can_return_help_message_with_models_info(bot: TransferBot):
    """Проверяем корректность обработки команды help."""
    message_mock = get_message_mock(text="help")
    
    # 1. проверяем, что используется нужный хэндлер и вызываем его
    actual_handler = await check_correct_handler(bot, message_mock, TransferBot.help_handler)
    await actual_handler(message_mock)

    # 2. проверяем, что отправлено сообщение с правильным текстом
    message_mock.reply.assert_called_once()
    assert "you need to send me any picture" in message_mock.reply.call_args_list[0].args[0], (
        "help message not found in reply of help handler."
    )

    # 3. проверяем, что были отправлены скриншоты с примерами стилей
    # и что их количество равно количеству имеющихся стилей
    reply_message_mock = message_mock.reply.return_value
    reply_message_mock.reply_media_group.assert_called()
    
    media_group: MediaGroup = reply_message_mock.reply_media_group.call_args_list[0]
    actual_models_count = len(media_group.media)
    expected_models_count = DEFAULT_MODELS_NUMBER

    assert actual_models_count == expected_models_count, (
        "count of examples isn't equal to real models number."
    )


async def test_bot_can_handle_unknown_messages(bot: TransferBot):
    """Проверяем, что бот умеет отвечать на "неизвестные" сообщения."""
    message_mock = get_message_mock(text="UNKNOWN_TEXT")
    
    # 1. проверяем, что используется нужный хэндлер и вызываем его
    actual_handler = await check_correct_handler(bot, message_mock, TransferBot.unknown_handler)
    await actual_handler(message_mock)

    # 2. проверяем, что отправлено сообщение с правильным текстом
    message_mock.reply.assert_called_once_with(unknown_message, parse_mode="MarkdownV2")


@pytest.mark.slow
@pytest.mark.parametrize("model_id", DEFAULT_MODEL_ID_LIST)
async def test_bot_can_process_content_photo_correctly(bot: TransferBot, model_id: str):
    """Проверяем, что бот корректно обрабатывает перенос подготовленного стиля."""
    message_mock = get_message_mock(photo=[PhotoSize(file_id=DEFAULT_USER_ID)])
    
    # 1. проверяем, что используется нужный хэндлер и вызываем его
    actual_handler = await check_correct_handler(bot, message_mock, TransferBot.process_content_photo)
    await actual_handler(message_mock)
    
    # эмулируем выбор модели пользователем и отправляем его в следующий хэндлер
    callback_query_mock = get_callback_query_mock(
        model_id=model_id,
        user_id=DEFAULT_USER_ID,
        message_id=message_mock.message_id,
    )
    
    # 2. проверяем, что  для обработки выбора стиля используется нужный хэндлер и вызываем его
    actual_query_handler = await check_correct_handler(
        bot, callback_query_mock, TransferBot.process_model_selection,
    )    
    await actual_query_handler(callback_query_mock)
    
    # 3. проверяем, что было отправлено сообщение с результатом
    bot.bot.send_photo.assert_called_once()
    kwargs = bot.bot.send_photo.call_args_list[0].kwargs
    assert_correct_result_photo(message_mock, kwargs, model_id)


@pytest.mark.slow
async def test_bot_can_process_content_and_your_style_photo_correctly(bot: TransferBot):
    """Проверяем, что бот корректно обрабатывает перенос пользовательского стиля."""
    message_mock = get_message_mock(photo=[PhotoSize(file_id=DEFAULT_USER_ID)])

    # 1. проверяем, что используется нужный хэндлер и вызываем его
    actual_handler = await check_correct_handler(bot, message_mock, TransferBot.process_content_photo)
    await actual_handler(message_mock)

    # эмулируем выбор модели пользователем и отправляем его в следующий хэндлер
    callback_query_mock = get_callback_query_mock(
        model_id=DEFAULT_YOUR_STYLE_MODEL_ID,
        user_id=DEFAULT_USER_ID,
        message_id=message_mock.message_id,
    )
    
    # 2. проверяем, что  для обработки выбора стиля используется нужный хэндлер и вызываем его
    actual_query_handler = await check_correct_handler(bot, callback_query_mock, TransferBot.process_model_selection)    
    await actual_query_handler(callback_query_mock)
    
    # 3. провряем, что для обработки фото со стилем используется правильный хэндлер
    message_mock.reply_to_message = get_message_mock(message_id=0)
    actual_handler = await check_correct_handler(bot, message_mock, TransferBot.process_style_photo)
    await actual_handler(message_mock)

    # 4. проверяем, что было отправлено сообщение с результатом
    bot.bot.send_photo.assert_called_once()
    kwargs = bot.bot.send_photo.call_args_list[0].kwargs
    assert_correct_result_photo(message_mock, kwargs, DEFAULT_YOUR_STYLE_MODEL_ID)


@pytest.mark.slow
async def test_bot_can_process_multiple_requests_at_the_same_time(bot: TransferBot):
    """Проверяем, что бот может корректно обрабатывать несколько запросов в один момент."""
    
    # 1. создаём сообщения и подготавливаем корутины
    messages = [
        get_message_mock(photo=[PhotoSize(file_id=DEFAULT_USER_ID)]) for _ in range(DEFAULT_NUMBER_OF_PARALLEL_REQUESTS)
    ]
    process_content_photo_handlers = []
    for message in messages:
        actual_handler = await check_correct_handler(bot, message, TransferBot.process_content_photo)
        process_content_photo_handlers.append(actual_handler(message))

    # асинхронно запускаем все обработчики
    await asyncio.gather(*process_content_photo_handlers)
    
    # 2. создаём выбор модели и подготавливаем корутины
    model_selection_handlers = []
    model_ids = []
    for message in messages:
        model_id = random.choice(DEFAULT_MODEL_ID_LIST)
        model_ids.append(model_id)
        callback_query_mock = get_callback_query_mock(
            model_id=model_id,
            user_id=DEFAULT_USER_ID,
            message_id=message.message_id,
        )
        actual_query_handler = await check_correct_handler(bot, callback_query_mock, TransferBot.process_model_selection)
        model_selection_handlers.append(actual_query_handler(callback_query_mock))
    
    # асинхронно запускаем все обработчики выбора модели
    await asyncio.gather(*model_selection_handlers)
    
    # 3. проверям, что бот ответил на все запросы
    assert len(model_ids) == bot.bot.send_photo.call_count
    for model_id, message, call_args in zip(model_ids, messages, bot.bot.send_photo.call_args_list):
        kwargs = call_args.kwargs
        assert_correct_result_photo(message, kwargs, model_id)
