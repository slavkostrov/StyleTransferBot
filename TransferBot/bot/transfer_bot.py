import asyncio
import logging
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

        self.queue = list()  # queue.Queue()

    def _setup_handlers(self):
        self.dispatcher.register_message_handler(self.send_welcome, commands=["start", "help"])
        self.dispatcher.register_message_handler(self.process_photo, content_types=["photo"])

    def run(self):
        executor.start_polling(self.dispatcher, skip_updates=True)

    async def send_welcome(self, message: types.Message):
        LOGGER.info(f"Sending welcome message to {message.chat.id}.")
        await message.reply(welcome_message.format(message=message), parse_mode='MarkdownV2')

    async def process_photo(self, message: types.Message):

        msg_hash = hash(message)
        self.queue.append(msg_hash)
        pos = self.queue.index(msg_hash) + 1
        reply_message = await message.reply(f"Ваше фото обрабатывается (место в очереди - {pos}).")
        current_position = pos

        file_obj = await self.bot.get_file(message.photo[-1].file_id)
        temp_file = BytesIO()
        await file_obj.download(temp_file)

        while True:
            LOGGER.info("цикл")
            new_current_position = self.queue.index(msg_hash) + 1
            if new_current_position != current_position:
                current_position = new_current_position
                reply_message = await reply_message.edit_text(f"Ваше фото {current_position} в очереди.")
            else:
                await asyncio.sleep(1)

            if current_position <= 2:
                reply_message = await reply_message.edit_text(f"Ваше фото обрабатывается.")
                break

        temp_file = await self.model.process_image(temp_file)

        await self.bot.delete_message(message.chat.id, reply_message.message_id)
        await self.bot.send_photo(
            chat_id=message.chat.id,
            photo=InputFile(temp_file),
            caption="Результат переноса стиля!",
            reply_to_message_id=message.message_id,
        )

        self.queue.remove(msg_hash)
        LOGGER.info(str(self.queue))


if __name__ == '__main__':
    bot = TransferBot(
        "",
        model=VGGTransfer(),
    )
    bot.run()
