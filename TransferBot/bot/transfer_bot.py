import logging
import queue
import typing as tp
from io import BytesIO
from logging import getLogger

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types.input_file import InputFile

from TransferBot.model import ModelABC
from TransferBot.model import VGGTransfer
from bot_answers import welcome_message

logging.basicConfig(level=logging.INFO)
LOGGER = getLogger(__file__)
LOGGER.setLevel(logging.INFO)


class TransferBot:

    def __init__(self, bot_token: str, model: tp.Optional[ModelABC] = None):
        self.model = model
        self.bot = Bot(token=bot_token)
        self.dispatcher = Dispatcher(self.bot)
        self._setup_handlers()

        self.queue = queue.Queue()

    def _setup_handlers(self):
        self.dispatcher.register_message_handler(self.send_welcome, commands=["start", "help"])
        self.dispatcher.register_message_handler(self.process_photo, content_types=["photo"])

    def run(self):
        executor.start_polling(self.dispatcher, skip_updates=True)

    async def send_welcome(self, message: types.Message):
        LOGGER.info(f"Sending welcome message to {message.chat.id}.")
        await message.reply(welcome_message.format(message=message), parse_mode='MarkdownV2')

    async def styly(self, image):
        m = self.model()
        image = m.process_image(image)
        return image

    async def process_photo(self, message: types.Message):
        # await self.queue.put(1)
        reply_message = await message.reply(f"Ваше фото обрабатывается (место в очереди - {self.queue.qsize()}).")

        file_obj = await self.bot.get_file(message.photo[-1].file_id)
        temp_file = BytesIO()
        await file_obj.download(temp_file)
        temp_file = await self.styly(temp_file)

        # self.queue.get()
        await self.bot.send_photo(
            chat_id=message.chat.id,
            photo=InputFile(temp_file),
            caption="Результат!"
        )


if __name__ == '__main__':
    bot = TransferBot(
        "TOKEN",
        model=VGGTransfer,
    )
    bot.run()
