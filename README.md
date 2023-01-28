# StyleTransferBot 🖼️

You can enjoy bot here - [@ThroughTheEyesOfArtistsBot](https://t.me/ThroughTheEyesOfArtistsBot)

* [Description](#description)
* [Install](#install)
* [Usage](#usage)
* [Deploy with Docker](#docker)
* [Advanced options](#advanced)
* [Tests](#tests)
* [Development](#dev)
* [Examples](#runtime)



<a name="description"><h2>Description ✉:</h2></a>

This package introduces implementation of Machine Learning Telegram bot, that contains transfer style techniques.
These techniques (include slow style transfer and pretrained fast transfer models) can be used for transferring any styles
onto user's images.

Package structure:
* `bot` package:
  * `TransferBot` - main class of bot implementation, encapsulate `aiogram` bot class, setup handlers etc.
* `model` package:
  * `slow_transfer` module - contains implementations of slow style transfer algorithm.
  * `fast_transfer` module - contains implementations of pretrained style transfer models.

The bot is written using the aiogram asynchronous library.
The bot contains handlers for arriving at the input of images.
Image processing takes place in a separate process so that the bot does not "freeze" when processing new messages.
The following entities are also provided:
- configurable queue - allows you to specify the maximum number of simultaneously running style transfer processes,
convenient to configure depending on resources (see max_tasks in [bot.py](TranferBot/bot/run.py));
- timeout and retries - implemented timeouts and retries in case
the process with style transfer hangs or "dies" for some reason (see timeout and n_retries in [bot.py](TranferBot/bot/run.py));
- request cache - implemented a cache for style transfer requests to the bot so that in case of an unexpected launch,
the bot can continue the user's dialogue (see on_setup, on_shutdown methods in [bot.py](TranferBot/bot/run.py)).

Examples are collected in the corresponding section of README.

<a name="install"><h3>Install 🔨:</h3></a>

To install package you need to clone repo and setup dependencies from requirements:

```shell
git clone https://github.com/slavkostrov/StyleTransferBot.git
cd StyleTransferBot
pip install -r requirements.txt
pip install -e .
```

<a name="usage"><h3>Usage example (see [run.py](run.py)):</h3></a>

```python
import os
from TransferBot.bot import TransferBot


if __name__ == "__main__":
    token = os.getenv("TG_STYLE_BOT_TOKEN")
    bot = TransferBot(
        bot_token=token,
        timeout_seconds=600,
        max_tasks=2
    )
    bot.run()
```

<a name="docker"><h3>Deploy with Docker 📦:</h3></a>

1. Clone this repo.
2. Write an `.env` file with your `TG_STYLE_BOT_TOKEN` in it.
3. Run `docker-compose up -d` and wait for the build to finish, run.py will be used in container,
so you can edit if you want to.

That's it. Enjoy your dockerized transfer style bot everywhere. 🚀

<a name="advanced"><h3>Advanced options 🧘:</h3></a>

* **Custom models**

You can write your own model, that will implement `TransferBot.model.ModelABC` interface.
After, you can register this model. See example bellow:

```python
from TransferBot.model import register_model, MunkModel

if __name__ == "__main__":
    register_model(MunkModel, model_id="custom_model")
    # run bot
```

* **Custom bot's messages**

You can simply change bot's answers style by edit module with all templates.
You can find it in [bot_answers.py](./TransferBot/bot/bot_answers.py).

After that you'll see your model in bot keyboard.

<a name="tests"><h3>Tests 🧪:</h3></a>

Package contains several tests, which you can find in `/tests` directory.
To run them, you just need to write a few commands:

```shell
pip install -r requirements.txt # setup env
pip install -e .                # install bot package
pip install pytest              # install pytest
pytest testing/                 # run tests
```

<a name="dev"><h3>Development 👨‍💻:</h3></a>

Before starting development please run:

```shell
pip install pre-commit
pre-commit install
```

for install linters hooks.

<a name="runtime"><h3>Runtime examples:</h3></a>
* TBD

## Links
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf)

## Contacts ☎️:

[@slavkostrov](https://t.me/slavkostrov)

## License 🪪:

[MIT](LICENSE)
