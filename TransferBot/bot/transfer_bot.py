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
from aiogram.types.input_file import InputFile

from TransferBot.model import VGGTransfer
from TransferBot.model.protocol import ModelABC
from bot_answers import welcome_message

logging.basicConfig(level=logging.INFO)
LOGGER = getLogger(__file__)
LOGGER.setLevel(logging.INFO)


def process_func(queue: multiprocessing.Queue, model_class: type, image: BytesIO):
    model = model_class()
    result = model.process_image(image)
    queue.put(result)
    sys.exit(0)


@dataclass
class RequestsQueue:
    """Очередь запросов на обработку фотографий."""
    _list: tp.List[int] = field(default_factory=list, repr=False)

    def put(self, message: types.Message):
        request_id = hash(message)
        self._list.append(request_id)

    def remove(self, message: types.Message):
        request_id = hash(message)
        if request_id in self._list:
            self._list.remove(request_id)

    def get_position(self, message: types.Message) -> int:
        request_id = hash(message)
        return self._list.index(request_id) + 1


class TransferBot:
    """Класс бота для переноса стиля."""

    def __init__(self, bot_token: str, model: ModelABC, max_tasks: int = 2):
        """
        Конструктор бота для переноса стиля.

        :param bot_token: Telegram токен бота.
        :param model: инстанс моделя для переноса стиля.
        :param max_tasks: максимальное количество асинхронных задач переноса.
        """
        if not isinstance(model, ModelABC):
            raise RuntimeError("Not valid model.")

        self.model = model
        self.bot = Bot(token=bot_token)
        self.max_tasks = max_tasks
        self.dispatcher = Dispatcher(self.bot)
        self._setup_handlers()

        self.queue = RequestsQueue()

    def _setup_handlers(self):
        """Выполняет установку всех обработчиков."""
        self.dispatcher.register_message_handler(self.send_welcome, commands=["start", "help"])
        self.dispatcher.register_message_handler(self.process_photo, content_types=["photo"])

    def run(self):
        """Запускает бота."""
        executor.start_polling(self.dispatcher, skip_updates=True)

    @staticmethod
    async def send_welcome(message: types.Message):
        """Отсправляет приветственное сообщение."""
        LOGGER.info(f"Sending welcome message to {message.chat.id}.")
        await message.reply(welcome_message.format(message=message), parse_mode='MarkdownV2')

    async def process_photo(self, message: types.Message):
        """Реализует логику по обработке фотографий."""
        reply_message = await self._wait_in_queue(message)
        input_image = await self.download_image(message)

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=process_func, args=(queue, self.model, input_image), )
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

        await self.bot.delete_message(message.chat.id, reply_message.message_id)
        await self.bot.send_photo(
            chat_id=message.chat.id,
            photo=InputFile(transformed_image),
            caption="Результат переноса стиля!",
            reply_to_message_id=message.message_id,
        )
        self.queue.remove(message)

    async def _wait_in_queue(self, message: types.Message):
        """Реализует ожидание в очереди на обработку фотографий."""
        self.queue.put(message)
        current_position = self.queue.get_position(message)
        reply_message = await message.reply(f"Ваше фото {current_position} в очереди.")

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

    async def download_image(self, message: types.Message):
        """Скачивает картинку в BytesIO."""
        file_obj = await self.bot.get_file(message.photo[-1].file_id)
        byte_stream = BytesIO()
        await file_obj.download(byte_stream)
        return byte_stream


if __name__ == '__main__':
    from secret import TOKEN
    bot = TransferBot(
        TOKEN,
        model=VGGTransfer,
    )
    bot.run()
