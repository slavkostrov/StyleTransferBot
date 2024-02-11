welcome_message = """
**Hi, \@{message.chat.username}\!**

My name is __ThroughTheEyesOfArtistsBot__ and i can transfer style to your image
Try to send an image and I will return the result
"""

reply_with_style_message = "Send your style image in a reply message"
choose_model_message = "Selected style \- {model_id}"
choose_style_message = "Select style\:"

error_message = "Oops, mistake 😅, please try again"
result_message = "Style transfer result \({request.model_id}\)\:"

queue_position_message = "Your photo is {current_position} in line\."

processing_message = "Your photo is being processed\."
own_style_message = "Your style 🎨"
unknown_message = "Send an image or use the commands \/start and \/help\."

help_message = """
_I can help you\!_

To run style transfer you need to send me any picture\.
After that you'll be able to choose style \(including your own\!\)

The following styles are currently available \(i will send you train images in next message\)\:
{description}

For transferring your own style you need to press `{own_style_message}` button\.
"""
