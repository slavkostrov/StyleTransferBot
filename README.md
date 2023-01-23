# StyleTransferBot

##### Usage example (see [run.py](run.py)):

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

##### Advanced options:
* Custom models

You can write your own model, that will implement `TransferBot.model.ModelABC` interface. 
After, you can register this model. See example bellow:

```python
from TransferBot.model import register_model, MunkModel

if __name__ == "__main__":
    register_model(MunkModel, model_id="custom_model")
    # run bot
```

After that you'll see your model in bot keyboard.

##### Tests:
* TBD

##### Runtime examples:
* TBD