welcome_message = """
**Привет, {message.chat.first_name}\!**

Меня зовут __TransferStyleBot__ и я умею переносить стиль на твои картинки\.
Попробуй отправить мне картинку и я верну тебе результат\.
"""

reply_with_style_message = "Пришлите стиль ответным сообщением\."
choose_model_message = "Выбрана модель {model_id}\."
choose_style_message = "Выберите стиль\:"

error_message = "Ошибка при обработке, попробуйте ещё раз\."
result_message = "Результат переноса стиля \({request.model_id}\)\:"

queue_position_message = "Ваше фото {current_position} в очереди\."
processing_message = "Ваше фото обрабатывается\."
own_style_message = "Ваш стиль"
unknown_message = (
    "Пришлите изображение либо воспользуйтесь командами \/start и \/help\."
)
