import asyncio
import datetime
import logging
import multiprocessing
import os.path
import pickle
import sys
import typing as tp
from dataclasses import dataclass, field
from io import BytesIO
from logging import getLogger
from queue import Empty

from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher.filters import Filter
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types.input_file import InputFile
from aiogram.utils.callback_data import CallbackData

from .bot_answers import welcome_message
from ..model import MODEL_REGISTRY, ModelABC, VGG19Transfer

logging.basicConfig(level=logging.INFO)
LOGGER = getLogger("transfer_bot.py")
LOGGER.setLevel(logging.INFO)


def _process_func(queue: multiprocessing.Queue, model_class: type, content_image: BytesIO,
                  style_image: tp.Optional[BytesIO] = None) -> tp.NoReturn:
    """Функция для запуска переноса стиля в отдельном процессе."""
    model: ModelABC = model_class()
    result: BytesIO = model.process_image(content_image, style_image)
    queue.put(result)
    sys.exit(0)


# TODO: LRU or smth with timeout
_CACHE = {}

RequestAction = CallbackData('r', 'model', 'message_id')


class StyleAnswerFilter(Filter):
    """Фильтр сообщений с картинкой-стилем."""

    async def check(self, message: types.Message) -> bool:
        input_message = message.reply_to_message
        if input_message is None:
            return False
        request: Request = Request.get_from_cache(message.chat.id, input_message.message_id)
        return request is not None


# OPTIONAL: use aiogram states but it seems like some linear dialog
# in this project dialog isn't linear,
# i mean that use can select model for previous images (while processing current) etc
# so far we use request with dumping backups.
@dataclass
class Request:
    """Класс запроса на перенос стиля."""
    chat_id: int
    message_id: int
    content_file_id: str = field(repr=False)

    style_message_id: tp.Optional[int] = field(default=None)
    style_file_id: tp.Optional[str] = field(default=None, repr=False)
    model_id: tp.Optional[str] = field(default=None)

    def __post_init__(self):
        self.put_in_cache()

    def put_in_cache(self, message_id: int = None) -> tp.NoReturn:
        message_id = message_id if message_id is not None else self.message_id
        LOGGER.info(f"putting in cache with key ({self.chat_id}, {message_id}) (cache size - {len(_CACHE) + 1}).")
        _CACHE[(str(self.chat_id), str(message_id))] = self

    @staticmethod
    def pop_from_cache(chat_id: int, message_id: int) -> "Request":
        LOGGER.info(f"pop from cache with key ({chat_id}, {message_id}) (cache size - {len(_CACHE) - 1}).")
        return _CACHE.pop((str(chat_id), str(message_id)))

    @staticmethod
    def get_from_cache(chat_id: int, message_id: int) -> "Request":
        LOGGER.info(f"get from cache with key ({chat_id}, {message_id}).")
        return _CACHE.get((str(chat_id), str(message_id)))

    def __hash__(self) -> int:
        return hash(f"{self.chat_id}_{self.message_id}_{self.content_file_id}_{self.style_file_id}")


@dataclass
class RequestsQueue:
    """Очередь запросов на на перенос стиля."""
    _list: tp.List[int] = field(default_factory=list, repr=False)

    def put(self, request: Request) -> tp.NoReturn:
        """Добавляет запрос в очередь."""
        LOGGER.info(f"Added to queue {request}.")
        request_id = hash(request)
        self._list.append(request_id)

    def remove(self, request: Request) -> tp.NoReturn:
        """Удаляет запрос из очереди."""
        request_id = hash(request)
        if request_id in self._list:
            LOGGER.info(f"Removed from queue {request}.")
            self._list.remove(request_id)

    def get_position(self, request: Request) -> int:
        """Возвращает позицию запроса в очереди."""
        request_id = hash(request)
        return self._list.index(request_id) + 1


class TransferBot:
    """Класс бота для переноса стиля."""

    def __init__(self, bot_token: str, timeout_seconds: int = 600, max_tasks: int = 2, max_retries_number: int = 3):
        """
        Конструктор бота для переноса стиля.

        :param bot_token: Telegram токен бота.
        :param max_tasks: максимальное количество асинхронных задач переноса.
        :param max_retries_number: максимальное количество попыток обработки изображения.
        """

        self.max_retries_number = max_retries_number
        self.bot = Bot(token=bot_token)
        self.max_tasks = max_tasks
        self.dispatcher = Dispatcher(self.bot)
        self._setup_handlers()

        self.timeout_seconds = timeout_seconds
        self.queue = RequestsQueue()

    def _setup_handlers(self) -> tp.NoReturn:
        """Выполняет установку всех обработчиков."""
        LOGGER.info("Setup handlers.")
        self.dispatcher.register_message_handler(self.send_welcome, commands=["start", "help"])
        self.dispatcher.register_message_handler(self.process_style_photo, StyleAnswerFilter(), content_types=["photo"])
        self.dispatcher.register_message_handler(self.process_content_photo, content_types=["photo"])
        self.dispatcher.register_callback_query_handler(self.process_model_selection)
        LOGGER.info("Handlers setup is ended.")

    def run(self) -> tp.NoReturn:
        """Запускает бота."""
        executor.start_polling(
            self.dispatcher,
            skip_updates=False,
            on_startup=self.on_startup,
            on_shutdown=self.on_shutdown
        )

    @staticmethod
    async def send_welcome(message: types.Message) -> tp.NoReturn:
        """Отсправляет приветственное сообщение."""
        LOGGER.info(f"Sending welcome message to {message.chat.id}.")
        await message.reply(welcome_message.format(message=message), parse_mode='MarkdownV2')

    async def process_model_selection(self, query: types.CallbackQuery) -> tp.NoReturn:
        """Реализует обработку выбора стиля/модели."""
        _, model_id, message_id = query.data.split(":")
        LOGGER.info(f"Got model selection for message_id={message_id} and {model_id}")
        request = Request.pop_from_cache(query.from_user.id, message_id)
        # TODO: add check if not found
        request.model_id = model_id

        LOGGER.info(f"Processing model selection for {request}.")
        if model_id == "OWN":
            style_request_message = await self.bot.edit_message_text(
                text=f"Пришлите стиль ответным сообщением.",
                reply_markup=None,
                message_id=query.message.message_id,
                chat_id=request.chat_id,
            )
            request.put_in_cache(style_request_message.message_id)
        else:
            await self.bot.edit_message_text(
                text=f"Выбрана модель {model_id}.",
                reply_markup=None,
                message_id=query.message.message_id,
                chat_id=request.chat_id,
            )
            await self.apply_style(request)
            await self.bot.delete_message(request.chat_id, query.message.message_id)

    async def apply_style(self, request: Request) -> tp.NoReturn:
        """Реализует применение модели и отправку результата пользователю."""
        reply_message = await self._wait_in_queue(request)

        queue = multiprocessing.Queue()
        # TODO: rewrite model getter maybe
        process_kwargs = {
            "model_class": MODEL_REGISTRY.get(request.model_id, VGG19Transfer),
            "content_image": await self.download_image(request.content_file_id),
            "queue": queue,
        }
        if request.style_file_id:
            process_kwargs["style_image"] = await self.download_image(request.style_file_id)

        LOGGER.info(f"Starting process with {process_kwargs}.")
        process = multiprocessing.Process(target=_process_func, kwargs=process_kwargs, )
        process.start()

        # TODO: improve timeout and n_retries
        # TODO: save current process to cache and restore it after bot reload
        start_time = datetime.datetime.now()
        n_retries = 0
        while True:
            transformed_image = None
            try:
                transformed_image = queue.get_nowait()
            except Empty:
                pass
            if transformed_image is not None:
                LOGGER.info("got result from child process in processify")
                break

            process_time = datetime.datetime.now() - start_time
            if process_time.seconds > self.timeout_seconds:
                LOGGER.error("Got timeout while processing photo... trying it again.")
                if n_retries < self.max_retries_number:
                    n_retries = n_retries + 1
                    process.kill()
                    process = multiprocessing.Process(target=_process_func, kwargs=process_kwargs, )
                    process.start()
                    LOGGER.warning(f"{n_retries + 1} retry attempt is started.")
                else:
                    LOGGER.error("Reached max_retries_number, image wont be processed.")
                    process.kill()

            if not process.is_alive():
                await reply_message.edit_text("Ошибка при обработке, попробуйте ещё раз.")
                self.queue.remove(request)
                # TODO: pop request from _CACHE maybe?
                raise Exception("process is dead")

            await asyncio.sleep(1)

        await self.bot.delete_message(request.chat_id, reply_message.message_id)
        LOGGER.info(f"Sending result of {request}.")
        result_message = f"Результат переноса стиля ({request.model_id if request.model_id != 'OWN' else 'Собственный стиль'}):"
        await self.bot.send_photo(
            chat_id=request.chat_id,
            photo=InputFile(transformed_image),
            caption=result_message,
            reply_to_message_id=request.message_id,
        )
        self.queue.remove(request)

    async def process_style_photo(self, message: types.Message) -> tp.NoReturn:
        """Обрабатывает фотографию со стилем."""
        LOGGER.info(f"Got style photo, message_id = {message.message_id}")
        input_message = message.reply_to_message
        request: Request = Request.pop_from_cache(message.chat.id, input_message.message_id)
        request.style_file_id = message.photo[-1].file_id
        LOGGER.info(f"Found input request for style - {request}.")
        await self.apply_style(request)

    async def process_content_photo(self, message: types.Message) -> tp.NoReturn:
        """Реализует логику по обработке фотографий."""
        request = Request(
            chat_id=message.chat.id,
            message_id=message.message_id,
            content_file_id=message.photo[-1].file_id,
        )
        LOGGER.info(f"Got content photo, request - {request}.")
        keyboard: InlineKeyboardMarkup = self.make_keyboard(request.message_id)
        await self.bot.send_message(
            chat_id=message.chat.id,
            text=f"Выберите стиль:",
            reply_markup=keyboard,
            reply_to_message_id=message.message_id,
        )

    async def _wait_in_queue(self, request: Request) -> types.Message:
        """Реализует ожидание в очереди на обработку фотографий."""
        self.queue.put(request)
        current_position = self.queue.get_position(request)
        reply_message = await self.bot.send_message(
            chat_id=request.chat_id,
            text=f"Ваше фото {current_position} в очереди.",
            reply_to_message_id=request.message_id,
        )

        while True:
            position = self.queue.get_position(request)
            if position != current_position:
                LOGGER.info(f"Request {request} is moved from {current_position} to {position}.")
                current_position = position
                reply_message = await reply_message.edit_text(f"Ваше фото {current_position} в очереди.")
            else:
                await asyncio.sleep(1)

            if current_position <= self.max_tasks:
                LOGGER.info(f"Start style transfering for {request}.")
                reply_message = await reply_message.edit_text(f"Ваше фото обрабатывается.")
                break

        return reply_message

    async def download_image(self, file_id: str) -> BytesIO:
        """Скачивает картинку в BytesIO."""
        LOGGER.info(f"Downloading file with id={file_id}.")
        file_obj = await self.bot.get_file(file_id)
        byte_stream = BytesIO()
        await file_obj.download(byte_stream)
        return byte_stream

    @staticmethod
    def make_keyboard(message_id: tp.Union[int, str]) -> InlineKeyboardMarkup:
        keyboard = InlineKeyboardMarkup()
        for model_id in MODEL_REGISTRY.keys():
            button = InlineKeyboardButton(
                model_id,
                callback_data=RequestAction.new(
                    model=model_id,
                    message_id=message_id,
                )
            )
            keyboard.insert(button)
        button = InlineKeyboardButton(
            "Ваш стиль",
            callback_data=RequestAction.new(
                model="OWN",
                message_id=message_id,
            )
        )
        keyboard.insert(button)
        return keyboard

    # TODO: write cache and other 'backups' to databaase
    # for now we use pkl file with cache
    # so we can continue processing selection after bot's reload
    # but we can't continue photo processing if it was started already
    async def on_startup(self, *args):
        """Логика выполняемая перед стартом бота."""
        if os.path.exists("_cache.pkl"):
            with open("_cache.pkl", "rb") as file:
                global _CACHE
                _CACHE = pickle.load(file)

    async def on_shutdown(self, *args):
        """Логика выполняемая при отключении бота."""
        LOGGER.info("Saving _CACHE to pkl.")
        with open("_cache.pkl", "wb") as file:
            pickle.dump(_CACHE, file)


def test_run():
    from .secret import TOKEN
    bot = TransferBot(TOKEN)
    bot.run()
