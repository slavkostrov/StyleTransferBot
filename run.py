import os

from TransferBot.bot import TransferBot

if __name__ == "__main__":
    token = os.getenv("TG_STYLE_BOT_TOKEN")
    if token is None:
        raise RuntimeError("TG_STYLE_BOT_TOKEN env variable doesn't exist.")
    timeout_seconds = int(os.getenv("TG_STYLE_BOT_TIMEOUT_SECONDS", 5000))
    max_tasks = int(os.getenv("TG_STYLE_BOT_MAX_TASKS", 2))
    max_retries_number = int(os.getenv("TG_STYLE_BOT_MAX_RETRIES_NUMBER", 2))
    slow_transfer_iters = int(os.getenv("TG_STYLE_BOT_SLOW_TRANSFER_ITERS", 500))
    bot = TransferBot(
        bot_token=token,
        timeout_seconds=timeout_seconds,
        max_tasks=max_tasks,
        max_retries_number=max_retries_number,
        slow_transfer_iters=slow_transfer_iters,
    )
    bot.run()
