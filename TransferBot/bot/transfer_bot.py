"""Main module with TG bot description."""
import asyncio
import datetime
import logging
import multiprocessing
import os.path
import pickle
import sys
import typing as tp
from dataclasses import dataclass, field
from functools import partial
from io import BytesIO
from logging import getLogger
from queue import Empty

from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher.filters import Filter
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types.input_file import InputFile
from aiogram.utils.callback_data import CallbackData

from .bot_answers import *
from ..model import MODEL_REGISTRY, ModelABC, VGG19Transfer

logging.basicConfig(level=logging.INFO)
LOGGER = getLogger("transfer_bot.py")
LOGGER.setLevel(logging.INFO)


def _process_func(
    queue: multiprocessing.Queue,
    model_class: tp.Type[ModelABC],
    content_image: BytesIO,
    style_image: tp.Optional[BytesIO] = None,
) -> tp.NoReturn:
    """Function for processing image in separate process.

    :param queue: queue object for storing image
    :param model_class: class of transfer style model
    :param content_image: image with content to stylize
    :param style_image: image with style
    :return: None
    """
    model: ModelABC = model_class()
    result: BytesIO = model.process_image(content_image, style_image)
    queue.put(result)
    sys.exit(0)


# TODO: LRU or smth with timeout
_CACHE: tp.Dict[tp.Any, "Request"] = {}

RequestAction = CallbackData("r", "model", "message_id")


class StyleAnswerFilter(Filter):
    """Filter messages with user's style."""

    async def check(self, message: types.Message) -> bool:
        input_message = message.reply_to_message
        if input_message is None:
            return False
        request: tp.Optional[Request] = Request.get_from_cache(
            message.chat.id, input_message.message_id
        )
        return request is not None


# OPTIONAL: use aiogram states but it seems like some linear dialog
# in this project dialog isn't linear,
# i mean that use can select model for previous images (while processing current) etc
# so far we use request with dumping backups.
@dataclass
class Request:
    """Class of image processing request."""

    chat_id: int
    message_id: int
    content_file_id: str = field(repr=False)

    style_message_id: tp.Optional[int] = field(default=None)
    style_file_id: tp.Optional[str] = field(default=None, repr=False)
    model_id: tp.Optional[str] = field(default=None)

    def __post_init__(self):
        self.put_in_cache()

    def put_in_cache(self, message_id: int = None) -> tp.NoReturn:
        """Putting request into global cache.

        :param message_id: id of target message.
        :return: None
        """
        message_id = message_id if message_id is not None else self.message_id
        LOGGER.info(
            f"putting in cache with key ({self.chat_id}, {message_id}) (cache size - {len(_CACHE) + 1})."
        )
        _CACHE[(str(self.chat_id), str(message_id))] = self

    @staticmethod
    def pop_from_cache(chat_id: int, message_id: int) -> "Request":
        """Pop request from global cache.

        :param chat_id: chat of request
        :param message_id: message of request
        :return: Request object
        """
        LOGGER.info(
            f"pop from cache with key ({chat_id}, {message_id}) (cache size - {len(_CACHE) - 1})."
        )
        return _CACHE.pop((str(chat_id), str(message_id)))

    @staticmethod
    def get_from_cache(chat_id: int, message_id: int) -> tp.Optional["Request"]:
        """Get request from global cache.

        :param chat_id: chat of request
        :param message_id: message of request
        :return: Request object
        """
        LOGGER.info(f"get from cache with key ({chat_id}, {message_id}).")
        return _CACHE.get((str(chat_id), str(message_id)))

    def __hash__(self) -> int:
        return hash(
            f"{self.chat_id}_{self.message_id}_{self.content_file_id}_{self.style_file_id}"
        )


@dataclass
class RequestsQueue:
    """Queue of requests."""

    _list: tp.List[int] = field(default_factory=list, repr=False)

    def put(self, request: Request) -> tp.NoReturn:
        """Add new request to queue.

        :param request: request instance
        :return: None
        """
        LOGGER.info(f"Added to queue {request}.")
        request_id = hash(request)
        self._list.append(request_id)

    def remove(self, request: Request) -> tp.NoReturn:
        """Remove object from queue.

        :param request: request instance to be removed.
        :return: None
        """
        request_id = hash(request)
        if request_id in self._list:
            LOGGER.info(f"Removed from queue {request}.")
            self._list.remove(request_id)

    def get_position(self, request: Request) -> int:
        """Get position of request in queue.

        :param request: request instance to check
        :return: position
        """
        request_id = hash(request)
        return self._list.index(request_id) + 1


class TransferBot:
    """Class of style transfer telegram Bot."""

    def __init__(
        self,
        bot_token: str,
        timeout_seconds: int = 10000,
        max_tasks: int = 2,
        max_retries_number: int = 3,
        slow_transfer_iters: int = 5000,
    ):
        """TransferBot constructor

        :param bot_token: telegram bot token.
        :param timeout_seconds: timeout seconds of image processing.
        :param max_tasks: number of maximum tasks for parallel processing.
        :param max_retries_number: maximum number of retries if image processing is failed.
        :param slow_transfer_iters: number of iterations for slow transfer.
        """

        self.max_retries_number = max_retries_number
        self.bot = Bot(token=bot_token)
        self.max_tasks = max_tasks
        self.dispatcher = Dispatcher(self.bot)
        self._setup_handlers()

        self.timeout_seconds: int = timeout_seconds
        self.queue: RequestsQueue = RequestsQueue()

        self.slow_transfer_iters: int = slow_transfer_iters

    def _setup_handlers(self) -> tp.NoReturn:
        """Setup input messages handlers.."""
        LOGGER.info("Setup handlers.")
        self.dispatcher.register_message_handler(
            self.send_welcome, commands=["start", "help"]
        )
        self.dispatcher.register_message_handler(
            self.process_style_photo, StyleAnswerFilter(), content_types=["photo"]
        )
        self.dispatcher.register_message_handler(
            self.process_content_photo, content_types=["photo"]
        )
        self.dispatcher.register_callback_query_handler(self.process_model_selection)
        LOGGER.info("Handlers setup is ended.")

    def run(self) -> tp.NoReturn:
        """Run bot pooling."""
        executor.start_polling(
            self.dispatcher,
            skip_updates=False,
            on_startup=self.on_startup,
            on_shutdown=self.on_shutdown,
        )

    @staticmethod
    async def send_welcome(message: types.Message) -> tp.NoReturn:
        """Send welcome message to user."""
        LOGGER.info(f"Sending welcome message to {message.chat.id}.")
        await message.reply(
            welcome_message.format(message=message), parse_mode="MarkdownV2"
        )

    async def process_model_selection(self, query: types.CallbackQuery) -> tp.NoReturn:
        """Process user's selection of model."""
        _, model_id, message_id = query.data.split(":")
        LOGGER.info(f"Got model selection for message_id={message_id} and {model_id}")
        request = Request.pop_from_cache(query.from_user.id, message_id)
        # TODO: add check if not found
        request.model_id = model_id

        LOGGER.info(f"Processing model selection for {request}.")
        if model_id == "OWN":
            style_request_message = await self.bot.edit_message_text(
                text=reply_with_style_message.format(query=query, model_id=model_id),
                reply_markup=None,
                message_id=query.message.message_id,
                chat_id=request.chat_id,
                parse_mode="MarkdownV2",
            )
            request.put_in_cache(style_request_message.message_id)
        else:
            await self.bot.edit_message_text(
                text=choose_model_message.format(query=query, model_id=model_id),
                reply_markup=None,
                message_id=query.message.message_id,
                chat_id=request.chat_id,
                parse_mode="MarkdownV2",
            )
            await self.apply_style(request)
            await self.bot.delete_message(request.chat_id, query.message.message_id)

    async def apply_style(self, request: Request) -> tp.NoReturn:
        """Apply style to user's image and send result."""
        reply_message = await self._wait_in_queue(request)

        queue: multiprocessing.Queue = multiprocessing.Queue()
        # TODO: rewrite model getter maybe
        model_id: str = request.model_id
        process_kwargs = {
            "model_class": MODEL_REGISTRY.get(
                model_id, partial(VGG19Transfer, num_steps=self.slow_transfer_iters)
            ),
            "content_image": await self.download_image(request.content_file_id),
            "queue": queue,
        }
        if request.style_file_id:
            process_kwargs["style_image"] = await self.download_image(
                request.style_file_id
            )

        LOGGER.info(f"Starting process with {process_kwargs}.")
        process = multiprocessing.Process(
            target=_process_func,
            kwargs=process_kwargs,
        )
        process.start()

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

            # TODO: fix, doesnt work yet
            process_time = datetime.datetime.now() - start_time
            if process_time.seconds > self.timeout_seconds or not process.is_alive():
                if process.is_alive():
                    process.kill()

                LOGGER.error("Got timeout while processing photo... trying it again.")
                if n_retries < self.max_retries_number:
                    n_retries = n_retries + 1
                    process.kill()
                    process = multiprocessing.Process(
                        target=_process_func,
                        kwargs=process_kwargs,
                    )
                    process.start()
                    LOGGER.warning(f"{n_retries + 1} retry attempt is started.")
                else:
                    LOGGER.error("Reached max_retries_number, image wont be processed.")
                    await reply_message.edit_text(error_message)
                    self.queue.remove(request)
                    # TODO: pop request from _CACHE maybe?
                    raise Exception("process is dead")

            await asyncio.sleep(1)

        await self.bot.delete_message(request.chat_id, reply_message.message_id)
        LOGGER.info(f"Sending result of {request}.")
        await self.bot.send_photo(
            chat_id=request.chat_id,
            photo=InputFile(transformed_image),
            caption=result_message.format(request=request),
            reply_to_message_id=request.message_id,
            parse_mode="MarkdownV2",
        )
        self.queue.remove(request)

    async def process_style_photo(self, message: types.Message) -> tp.NoReturn:
        """Process style image."""
        LOGGER.info(f"Got style photo, message_id = {message.message_id}")
        input_message = message.reply_to_message
        request: Request = Request.pop_from_cache(
            message.chat.id, input_message.message_id
        )
        request.style_file_id = message.photo[-1].file_id
        LOGGER.info(f"Found input request for style - {request}.")
        await self.apply_style(request)

    async def process_content_photo(self, message: types.Message) -> tp.NoReturn:
        """Process content images."""
        request = Request(
            chat_id=message.chat.id,
            message_id=message.message_id,
            content_file_id=message.photo[-1].file_id,
        )
        LOGGER.info(f"Got content photo, request - {request}.")
        keyboard: InlineKeyboardMarkup = self.make_keyboard(request.message_id)
        await self.bot.send_message(
            chat_id=message.chat.id,
            text=choose_style_message.format(message=message),
            reply_markup=keyboard,
            reply_to_message_id=message.message_id,
            parse_mode="MarkdownV2",
        )

    async def _wait_in_queue(self, request: Request) -> types.Message:
        """Put requests into queue and monitor it."""
        self.queue.put(request)
        current_position = self.queue.get_position(request)
        reply_message = await self.bot.send_message(
            chat_id=request.chat_id,
            text=queue_position_message.format(current_position=current_position),
            reply_to_message_id=request.message_id,
            parse_mode="MarkdownV2",
        )

        while True:
            position = self.queue.get_position(request)
            if position != current_position:
                LOGGER.info(
                    f"Request {request} is moved from {current_position} to {position}."
                )
                current_position = position
                reply_message = await reply_message.edit_text(
                    queue_position_message.format(current_position=current_position),
                    parse_mode="MarkdownV2",
                )
            else:
                await asyncio.sleep(1)

            if current_position <= self.max_tasks:
                LOGGER.info(f"Start style transfering for {request}.")
                reply_message = await reply_message.edit_text(
                    processing_message.format(request=request),
                    parse_mode="MarkdownV2",
                )
                break

        return reply_message

    async def download_image(self, file_id: str) -> BytesIO:
        """Download image into BytesIO stream."""
        LOGGER.info(f"Downloading file with id={file_id}.")
        file_obj = await self.bot.get_file(file_id)
        byte_stream = BytesIO()
        await file_obj.download(byte_stream)
        return byte_stream

    @staticmethod
    def make_keyboard(message_id: tp.Union[int, str]) -> InlineKeyboardMarkup:
        """Make keyboard based on model registry."""
        keyboard = InlineKeyboardMarkup()
        for model_id in MODEL_REGISTRY.keys():
            button = InlineKeyboardButton(
                model_id,
                callback_data=RequestAction.new(
                    model=model_id,
                    message_id=message_id,
                ),
            )
            keyboard.insert(button)
        button = InlineKeyboardButton(
            own_style_message,
            callback_data=RequestAction.new(
                model="OWN",
                message_id=message_id,
            ),
        )
        keyboard.insert(button)
        return keyboard

    # TODO: write cache and other 'backups' to databaase
    # for now we use pkl file with cache
    # so we can continue processing selection after bot's reload
    # but we can't continue photo processing if it was started already
    async def on_startup(self, *args):
        """Startup preparation.
        Load previous requests from local cache"""
        if os.path.exists("_cache.pkl"):
            with open("_cache.pkl", "rb") as file:
                global _CACHE
                _CACHE = pickle.load(file)

    async def on_shutdown(self, *args):
        """Shutdown preparation.
        Save active (w/o answers) requests to cache."""
        LOGGER.info("Saving _CACHE to pkl.")
        with open("_cache.pkl", "wb") as file:
            pickle.dump(_CACHE, file)
