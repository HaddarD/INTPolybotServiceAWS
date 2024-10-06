import uuid
import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
from img_proc import Img
from collections import Counter
import json


class Bot:

    def __init__(self, token, telegram_chat_url):
        self.image_path = ""
        self.images = []
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)
        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)
        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)
        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')


    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    @staticmethod
    def is_current_msg_photo(msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')
        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)
        return file_info.file_path

    def send_photo(self, chat_id, image_path, caption=None):
        if not os.path.exists(image_path):
            raise RuntimeError("Image path doesn't exist")
        if caption is None:
            self.telegram_bot_client.send_photo(
                chat_id,
                InputFile(image_path)
            )
        else:
            self.telegram_bot_client.send_photo(
                chat_id,
                InputFile(image_path),
                caption=caption
            )

    def handle_message(self, msg):
        """
        Determines what to do based on the message received from the user:
        Ignore files without a specific processing function.
        Reply to any text message with a quote + the user name.
        If the message has a photo:
        Determines whether there's 1 or 2 images & call the menu function accordingly.
        """
        logger.info(f'Incoming message: {msg}')
        chat_id = msg['chat']['id']
        try:
            if 'text' in msg:
                if 'entities' in msg and msg['entities'][0]['type'] == 'bot_command':
                    self.handle_filter_command(msg)
                elif msg["text"] != 'Please don\'t quote me':
                    self.send_text_with_quote(
                        chat_id,
                        f"Hey! <('-'<)<('.')>(>'-')> \n {msg['text']}",
                        quoted_msg_id=msg['message_id']
                    )
                else:
                    self.send_text(chat_id, "Hey :D")
            elif 'document' in msg:
                pass
            elif 'photo' in msg:
                chat_id = msg['chat']['id']
                if 'media_group_id' in msg:
                    self.image_path = self.download_user_photo(msg)
                    self.images.append(self.image_path)
                    if len(self.images) == 2:
                        self.send_photo_command_menu(chat_id)
                else:
                    self.image_path = self.download_user_photo(msg)
                    self.send_photo_command_menu(chat_id)
            elif 'audi' in msg:
                pass
            elif 'voice' in msg:
                pass
            else:
                pass
        except Exception:
            self.send_text(chat_id, "Please try again")

    def send_photo_command_menu(self, chat_id):
        """
        Sends a links menu of the available filters as a response message to any image or images group received from the user
        """
        command_menu = "Please reply with the desired filter command:\n"
        command_menu += "/blur - Apply blur filter\n"
        command_menu += "/contour - Apply contour filter\n"
        command_menu += "/rotate - Rotate the image\n"
        command_menu += "/salt_n_pepper - Apply salt and pepper noise\n"
        command_menu += "/concat - Requires 2 Images to be uploaded, & collages them together\n"
        command_menu += "/segment - Segment the image\n"
        command_menu += "/detect - Identify the image content using YOLOv5\n"
        self.send_text(chat_id, command_menu)

    def send_photo_command_submenu(self, chat_id):
        """
        Sends a links submenu of directions options when the user clicks the concat filter option
        """
        command_submenu = "Please reply with the desired direction:\n"
        command_submenu += "/horizontal\n"
        command_submenu += "/vertical\n"
        self.send_text(chat_id, command_submenu)

    def handle_filter_command(self, msg):
        """
        Handles the user's choice of a filter by calling the corresponding function to the filter command
        """
        chat_id = msg['chat']['id']
        command = msg['text'].split()[0]
        error_found = False
        processed_image_path = ""
        if command == '/concat':
            self.send_photo_command_submenu(chat_id)
        elif command == '/horizontal' or command == '/vertical':
            if len(self.images) < 2:
                self.send_text(chat_id, "Please upload 2 images together to concat.")
                return
            elif self.images:
                img1 = Img(self.images[0])
                img2 = Img(self.images[1])
                result = img1.concat(img2, command)
                if result[1] == 500:
                    self.send_text(chat_id, result[0])
                    self.image_path = ""
                    self.images = []
                    return
                processed_image_path = img1.save_img()
            else:
                error_found = True
        elif self.image_path:
            img = Img(self.image_path)
            if command == '/blur':
                img.blur()
            elif command == '/contour':
                img.contour()
            elif command == '/rotate':
                img.rotate()
            elif command == '/salt_n_pepper':
                img.salt_n_pepper()
            elif command == '/segment':
                img.segment()
            elif command == '/detect':
                try:
                    self.send_text(chat_id, "Your image is being processed. Please wait...")
                    image_path = os.path.abspath(self.image_path)
                    image_name = os.path.basename(self.image_path)
                    detection_result, status_code = img.upload_and_detect(image_path, image_name, chat_id)
                    logger.info(f"Detection result: {detection_result}, status code: {status_code}")
                    # if isinstance(detection_result, tuple) and len(detection_result) == 2:
                    #     detection_summary, status_code = detection_result
                    if detection_result.get('status') == 'error':
                        self.send_text(chat_id, f"Sorry! >_< There was a Detection error: {detection_result['error_message']}")
                        return
                    else:
                        detection_summary, status_code = detection_result, 200
                    if status_code != 200:
                        error_message = detection_summary if isinstance(detection_summary, str) else "An error occurred during detection."
                        logger.warning(f"Sending error message to user: {error_message}")
                        self.send_text(chat_id, error_message)
                        return
                    if detection_summary:
                        processed_image_path = detection_summary.get('processed_image_path', [])
                        caption = self.detection_decode(detection_summary)
                        self.send_photo(chat_id, processed_image_path, caption)
                        self.images = []
                        self.image_path = ""
                    else:
                        self.send_text(chat_id, "No objects detected in the image.")
                    return
                except Exception as e:
                    logger.error(f"Error during prediction: {e}", exc_info=True)
                    self.send_text(chat_id, ">_< Sorry! An unexpected error occurred during detection. Please try again. >_<")
                    return
            else:
                # Invalid filter command
                self.send_text(chat_id, "Invalid filter command. Please choose from the command menu.")
                return
            processed_image_path = img.save_img()
        if error_found is False and processed_image_path:
            self.send_photo(chat_id, processed_image_path)
            self.image_path = ""
            self.images = []

    @staticmethod
    def detection_decode(detection_summary):
        if detection_summary:
            with open(detection_summary['processed_image_path'], 'rb') as photo:
                try:
                    # Check the type of prediction_summary and log it
                    logger.info(f"Detection summary type: {type(detection_summary)}")
                    logger.info(f"Detection summary content: {detection_summary}")
                    # Assuming prediction_summary is a list of dictionaries
                    if isinstance(detection_summary, dict) and 'result_path' in detection_summary:
                        result_path = detection_summary['result_path']
                        if 'labels' in result_path:
                            labels = result_path['labels']
                            classes = [label['class'] for label in labels]
                        else:
                            raise KeyError("Missing 'labels' in 'result_path'")
                    else:
                        raise KeyError("Invalid structure for prediction_summary")
                    quantities = Counter(classes)
                    response = "Detection Summary:\n"
                    response += "\n".join([f"{key.capitalize()} - {value}" for key, value in quantities.items()])
                    return response
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.error(f'Error decoding prediction summary: {e}')
                    return "Error decoding prediction summary"

            # TODO upload the photo to S3 """(in img_proc)"""
            # TODO send a job to the SQS queue """(in img_proc)"""
            # TODO send message to the Telegram end-user (e.g. Your image is being processed. Please wait...) """(in the handle_filter_command)"""
