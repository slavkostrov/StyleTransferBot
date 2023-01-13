import asyncio
import logging
import multiprocessing
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

from TransferBot.model import MODEL_REGISTRY
from bot_answers import welcome_message

logging.basicConfig(level=logging.INFO)
LOGGER = getLogger(__file__)
LOGGER.setLevel(logging.INFO)


def process_func(queue: multiprocessing.Queue, model_class: type, content_image: BytesIO,
                 style_image: tp.Optional[BytesIO] = None) -> tp.NoReturn:
    model: "ModelABC" = model_class()
    result: BytesIO = model.process_image(content_image
                                          , style_image)
    queue.put(result)
    sys.exit(0)


_CACHE = {}

RequestAction = CallbackData('r', 'model', 'message_id')


class StyleAnswerFilter(Filter):
    async def check(self, message: types.Message) -> bool:  # [3]
        input_message = message.reply_to_message
        if input_message is None:
            return False
        request: Request = Request.get_from_cache(message.chat.id, input_message.message_id)
        return request is not None


@dataclass
class Request:
    chat_id: int
    message_id: int
    content_file_id: str = field(repr=False)

    style_message_id: tp.Optional[int] = field(default=None)
    style_file_id: tp.Optional[str] = field(default=None, repr=False)
    model_id: tp.Optional[str] = field(default=None)

    def __post_init__(self):
        self.put_in_cache()

    def put_in_cache(self, message_id: int = None) -> tp.NoReturn:
        LOGGER.info(f"putting in cache with key ({self.chat_id}, {message_id}) (cache size - {len(_CACHE) + 1}).")
        message_id = message_id if message_id is not None else self.message_id
        _CACHE[(str(self.chat_id), str(message_id))] = self

    @staticmethod
    def pop_from_cache(chat_id: int, message_id: int) -> "Request":
        LOGGER.info(f"pop from cache with key ({chat_id}, {message_id}) (cache size - {len(_CACHE) - 1}).")
        return _CACHE.pop((str(chat_id), str(message_id)))

    @staticmethod
    def get_from_cache(chat_id: int, message_id: int) -> "Request":
        LOGGER.info(f"get from cache with key ({chat_id}, {message_id}).")
        return _CACHE.get((str(chat_id), str(message_id)))

    def __hash__(self):
        return hash(f"{self.chat_id}_{self.message_id}_{self.content_file_id}_{self.style_file_id}")


@dataclass
class RequestsQueue:
    """Очередь запросов на обработку фотографий."""
    _list: tp.List[int] = field(default_factory=list, repr=False)

    def put(self, request: Request) -> tp.NoReturn:
        LOGGER.info(f"Added to queue {request}.")
        request_id = hash(request)
        self._list.append(request_id)

    def remove(self, request: Request) -> tp.NoReturn:
        request_id = hash(request)
        if request_id in self._list:
            LOGGER.info(f"Removed from queue {request}.")
            self._list.remove(request_id)

    def get_position(self, request: Request) -> int:
        request_id = hash(request)
        return self._list.index(request_id) + 1


class TransferBot:
    """Класс бота для переноса стиля."""

    def __init__(self, bot_token: str, max_tasks: int = 2):
        """
        Конструктор бота для переноса стиля.

        :param bot_token: Telegram токен бота.
        :param max_tasks: максимальное количество асинхронных задач переноса.
        """

        self.bot = Bot(token=bot_token)
        self.max_tasks = max_tasks
        self.dispatcher = Dispatcher(self.bot)
        self._setup_handlers()

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
        executor.start_polling(self.dispatcher, skip_updates=True)

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
        request.model_id = model_id

        LOGGER.info(f"Processing model selection for {request}.")
        if model_id == "OWN_STYLE":
            style_request_message = await self.bot.edit_message_text(
                text=f"Выбран собственный стиль, пришлите стиль ответным сообщением.",
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
        # TODO: rewrite model getter maybe
        reply_message = await self._wait_in_queue(request)

        queue = multiprocessing.Queue()
        process_kwargs = {
            "model_class": MODEL_REGISTRY.get(request.model_id, MODEL_REGISTRY["VGG16"]),
            "content_image": await self.download_image(request.content_file_id),
            "queue": queue,
        }
        if request.style_file_id:
            process_kwargs["style_image"] = await self.download_image(request.style_file_id)

        LOGGER.info(f"Starting process with {process_kwargs}.")
        process = multiprocessing.Process(target=process_func, kwargs=process_kwargs, )
        process.start()

        # TODO: add something like timeout
        while True:
            transformed_image = None
            try:
                transformed_image = queue.get_nowait()
            except Empty:
                pass
            if transformed_image is not None:
                LOGGER.info("got result from child process in processify")
                break
            if not process.is_alive():
                raise Exception("process is dead")
            await asyncio.sleep(1)

        await self.bot.delete_message(request.chat_id, reply_message.message_id)
        LOGGER.info(f"Sending result of {request}.")
        await self.bot.send_photo(
            chat_id=request.chat_id,
            photo=InputFile(transformed_image),
            caption="Результат переноса стиля" + (
                f"c использованием {request.model_id}!" if (request.model_id != "OWN_STYLE") else "!"),
            reply_to_message_id=request.message_id,
        )
        self.queue.remove(request)

    async def process_style_photo(self, message: types.Message) -> tp.NoReturn:
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
            text=f"Выберите модель:",
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
            "Свой стиль",
            callback_data=RequestAction.new(
                model="OWN_STYLE",
                message_id=message_id,
            )
        )
        keyboard.insert(button)
        return keyboard


if __name__ == '__main__':
    from secret import TOKEN

    bot = TransferBot(TOKEN)
    bot.run()
