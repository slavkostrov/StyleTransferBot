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
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types.input_file import InputFile
from aiogram.utils.callback_data import CallbackData

from TransferBot.model import MODEL_REGISTRY
from bot_answers import welcome_message

logging.basicConfig(level=logging.INFO)
LOGGER = getLogger(__file__)
LOGGER.setLevel(logging.INFO)


def process_func(queue: multiprocessing.Queue, model_class: type, image: BytesIO):
    model = model_class()
    result = model.process_image(image)
    queue.put(result)
    sys.exit(0)


_CACHE = {}

RequestAction = CallbackData('r', 'model', 'request_id')


@dataclass
class Request:
    chat_id: int
    message_id: int
    content_file_id: tp.Union[int, tp.List[int]]
    style_file_id: tp.Union[int, tp.List[int], None] = field(default=None)

    def __post_init__(self):
        _CACHE[str(hash(self))] = self

    def __hash__(self):
        return hash(f"{self.chat_id}_{self.message_id}_{self.content_file_id}_{self.style_file_id}")

    # TODO: timeout and delete messages after it
    # def __del__


@dataclass
class RequestsQueue:
    """Очередь запросов на обработку фотографий."""
    _list: tp.List[int] = field(default_factory=list, repr=False)

    def put(self, message: tp.Any):
        request_id = hash(message)
        self._list.append(request_id)

    def remove(self, message: tp.Any):
        request_id = hash(message)
        if request_id in self._list:
            self._list.remove(request_id)

    def get_position(self, message: tp.Any) -> int:
        request_id = hash(message)
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

    def _setup_handlers(self):
        """Выполняет установку всех обработчиков."""
        self.dispatcher.register_message_handler(self.send_welcome, commands=["start", "help"])
        self.dispatcher.register_message_handler(self.process_photo, content_types=["photo"])
        self.dispatcher.register_callback_query_handler(self.apply_style)

    def run(self):
        """Запускает бота."""
        executor.start_polling(self.dispatcher, skip_updates=True)

    @staticmethod
    async def send_welcome(message: types.Message):
        """Отсправляет приветственное сообщение."""
        LOGGER.info(f"Sending welcome message to {message.chat.id}.")
        await message.reply(welcome_message.format(message=message), parse_mode='MarkdownV2')

    async def apply_style(self, query: types.CallbackQuery):
        _, model_id, request_id = query.data.split(":")
        request = _CACHE.pop(request_id)

        await self.bot.edit_message_text(
            text=f"Выбрана модель {model_id}.",
            reply_markup=None,
            message_id=query.message.message_id,
            chat_id=request.chat_id,
        )

        model = MODEL_REGISTRY.get(model_id)  # TODO: FIX, make model registry
        reply_message = await self._wait_in_queue(request.chat_id, request.message_id)
        input_image = await self.download_image(request.content_file_id)

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=process_func, args=(queue, model, input_image), )
        process.start()

        # TODO: add something like timeout
        while True:
            data = None
            try:
                data = queue.get_nowait()
            except Empty:
                pass
            if data is not None:
                LOGGER.info("got result from child process in processify")
                transformed_image = data
                break
            if not process.is_alive():
                raise Exception("process is dead")
            await asyncio.sleep(1)

        await self.bot.delete_message(request.chat_id, reply_message.message_id)
        await self.bot.delete_message(request.chat_id, query.message.message_id)
        await self.bot.send_photo(
            chat_id=request.chat_id,
            photo=InputFile(transformed_image),
            caption=f"Результат переноса стиля c использованием {model_id}!",
            reply_to_message_id=request.message_id,
        )
        self.queue.remove(request.message_id)

    async def process_photo(self, message: types.Message):
        """Реализует логику по обработке фотографий."""
        request = Request(
            chat_id=message.chat.id,
            message_id=message.message_id,
            content_file_id=message.photo[-1].file_id,
        )
        keyboard: InlineKeyboardMarkup = self.make_keyboard(hash(request))
        await self.bot.send_message(
            chat_id=message.chat.id,
            text=f"Выберите модель:",
            reply_markup=keyboard,
            reply_to_message_id=message.message_id,
        )

    async def _wait_in_queue(self, chat_id: int, message: int):
        """Реализует ожидание в очереди на обработку фотографий."""
        self.queue.put(message)
        current_position = self.queue.get_position(message)
        reply_message = await self.bot.send_message(
            chat_id=chat_id,
            text=f"Ваше фото {current_position} в очереди.",
            reply_to_message_id=message,
        )

        while True:
            position = self.queue.get_position(message)
            if position != current_position:
                current_position = position
                reply_message = await reply_message.edit_text(f"Ваше фото {current_position} в очереди.")
            else:
                await asyncio.sleep(1)

            if current_position <= self.max_tasks:
                reply_message = await reply_message.edit_text(f"Ваше фото обрабатывается.")
                break

        return reply_message

    async def download_image(self, file_id: str):
        """Скачивает картинку в BytesIO."""
        file_obj = await self.bot.get_file(file_id)
        byte_stream = BytesIO()
        await file_obj.download(byte_stream)
        return byte_stream

    @staticmethod
    def make_keyboard(request_id: int):
        keyboard = InlineKeyboardMarkup()
        button = InlineKeyboardButton(
            'VGG16!',
            callback_data=RequestAction.new(
                model="VGGTransfer",
                request_id=request_id,
            )
        )
        keyboard.insert(button)
        button = InlineKeyboardButton(
            'VGG19!',
            callback_data=RequestAction.new(
                model="VGG19Transfer",
                request_id=request_id,
            )
        )
        keyboard.insert(button)
        return keyboard


if __name__ == '__main__':
    from secret import TOKEN
    bot = TransferBot(TOKEN)
    bot.run()
