from pathlib import Path
from matplotlib.image import imread, imsave
import random
import os
from loguru import logger
import boto3
from botocore.exceptions import ClientError
import json
import time
from boto3.dynamodb.types import TypeDeserializer
import threading
from sqs_listener import SQSListener


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


class Img:

    def __init__(self, path):
        self.path = Path(path)
        self.data = rgb2gray(imread(path)).tolist()
        self.bucket_name = os.getenv('BUCKET_NAME')
        self.sqs_client = boto3.client('sqs', region_name='us-east-1')
        self.job_queue_url = os.environ.get('SQS_JOB_QUEUE_URL')
        self.completion_queue_url = os.environ.get('SQS_COMPLETION_QUEUE_URL')
        self.dynamo_client = boto3.resource('dynamodb', region_name='us-east-1')
        self.table = self.dynamo_client.Table('DYNAMO_DB_NAME')
        self.sqs_listener = SQSListener(self.sqs_client, self.completion_queue_url, self.bucket_name)
        self.detection_results = {}
        # self.listener_thread = None

        if not self.job_queue_url or not self.completion_queue_url:
            raise EnvironmentError("SQS job or completion queue URL not set in environment variables.")

        self.listen_for_completion()

    def save_img(self):
        new_path = self.path.with_name(self.path.stem + '_filtered' + self.path.suffix)
        imsave(new_path, self.data, cmap='gray')
        return new_path

    def blur(self, blur_level=16):
        """
        Applies a blur filter to the image and saves it to send back to the user
        """
        height = len(self.data)
        width = len(self.data[0])
        filter_sum = blur_level ** 2
        result = []
        for i in range(height - blur_level + 1):
            row_result = []
            for j in range(width - blur_level + 1):
                sub_matrix = [row[j:j + blur_level] for row in self.data[i:i + blur_level]]
                average = sum(sum(sub_row) for sub_row in sub_matrix) // filter_sum
                row_result.append(average)
            result.append(row_result)
        self.data = result

    def contour(self):
        """
        Applies a contour filter to the image and saves it to send back to the user
        """
        for i, row in enumerate(self.data):
            res = []
            for j in range(1, len(row)):
                res.append(abs(row[j-1] - row[j]))
            self.data[i] = res

    def rotate(self):
        """
        Rotates the image and saves it to send back to the user
        """
        if not self.data:
            raise RuntimeError("Image data is empty")
        self.data = [list(row) for row in zip(*self.data[::-1])]

    def salt_n_pepper(self, salt_prob=0.05, pepper_prob=0.05):
        """
        Applies a salt & pepper filter on the image and saves it to send back to the user
        """
        height = len(self.data)
        width = len(self.data[0])
        for i in range(height):
            for j in range(width):
                rand = random.random()
                if rand < salt_prob:
                    self.data[i][j] = 255
                elif rand < salt_prob + pepper_prob:
                    self.data[i][j] = 0

    def concat(self, other_img, direction='/horizontal'):
        """
        merges 2 images into a collage either horizontally or vertically according to the user's choice and saves it to send back to the user
        """
        if direction == '/horizontal':
            try:
                if len(self.data) != len(other_img.data):
                    raise RuntimeError("Images must have the same height for horizontal concatenation")
                self.data = [row1 + row2 for row1, row2 in zip(self.data, other_img.data)]
            except RuntimeError as e:
                error_message = str(e)
                return error_message, 500
        elif direction == '/vertical':
            try:
                if len(self.data[0]) != len(other_img.data[0]):
                    raise RuntimeError("Images must have the same width for vertical concatenation")
                self.data += other_img.data
            except RuntimeError as e:
                error_message = str(e)
                return error_message, 500
        else:
            return "Invalid direction for concatenation. Must be 'horizontal' or 'vertical'.", 500
        return "Ok", 200

    def segment(self):
        """
        Applies a segment filter on the image and saves it to send back to the user
        """
        if not self.data:
            raise RuntimeError("Image data is empty")
        total_pixels = sum(sum(row) for row in self.data)
        average = total_pixels // (len(self.data) * len(self.data[0]))
        for i, row in enumerate(self.data):
            self.data[i] = [0 if pixel < average else 255 for pixel in row]

    def upload_and_detect(self, image_path, chat_id, detection_id, image_name=None):
        """
        Uploads the image to S3 & sends Yolo5 an SQS job request to detect the image content
        """
        if not image_path:
            raise ValueError("Image path is empty")
        if image_name is None:
            image_name = os.path.basename(image_path)
        try:
            self.upload_to_s3(image_path, image_name)
            logger.info(f"Successfully uploaded {image_name} to S3")
            self.send_job_to_sqs(image_name, chat_id, detection_id)
            logger.info(f"Job sent to YOLO5 for Detection ID: {detection_id}")
            self.detection_results[detection_id] = {'chat_id': chat_id, 'status': 'processing'}
            return "Processing started", 200
        except Exception as e:
            logger.exception(f'<red>Error uploading image to S3: {e}</red>')
            return str(e), 500

        logger.info(f"Starting detection on image: {image_name}")

        if isinstance(results, str) and "TimedOut" in results:
            return results, 500
        if not results:
            return "No detection results received.", 404
        logger.info(f"Results received for job {chat_id}: {results}")
        # Return the results back to bot.py for further processing
        return results, 200


    def upload_to_s3(self, image_path, image_name):
        s3_client = boto3.client('s3')
        s3_key_with_prefix = f"user_images/{image_name}"
        try:
            s3_client.upload_file(image_path, self.bucket_name, s3_key_with_prefix)
            return
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {e}")
            raise

    def send_job_to_sqs(self, image_name, chat_id, detection_id):
        message_body = json.dumps({
            'JobID': str(detection_id),
            'image_name': image_name,
            'chat_id': chat_id
        })

        try:
            self.sqs_client.send_message(
                QueueUrl=self.job_queue_url,
                MessageBody=message_body,
            )
            logger.info(f"Job sent to YOLO5 for Detection ID: {detection_id}")
        except ClientError as e:
            logger.error(f"Failed to send message to SQS: {e}")
            raise

    def listen_for_completion(self):
        self.sqs_listener.start_listening()
        # if not self.listener_thread or not self.listener_thread.is_alive():
        #     self.listener_thread = threading.Thread(target=self.sqs_listener.start_listening)
        #     self.listener_thread.daemon = True
        #     self.listener_thread.start()

        # max_attempts = 3  # Adjust as needed
        # # attempt = 0
        # # Poll SQS for completion messages
        # for attempt in range(max_attempts):
        # # while attempt < max_attempts:
        #     logger.info(f"Polling for completion for Detection ID: {detection_id}")
        #     try:
        #         response = self.sqs_client.receive_message(
        #             QueueUrl=self.completion_queue_url,
        #             AttributeNames=['All'],
        #             MessageAttributeNames=['All'],
        #             MaxNumberOfMessages=1,
        #             WaitTimeSeconds=10
        #         )
        #         logger.debug(f"SQS Response: {response}")
        #         if 'Messages' not in response or not response['Messages']:
        #             logger.info(f"No messages received for Detection ID: {detection_id}, continuing to poll")
        #             attempt += 1
        #             time.sleep(10)
        #             continue
        #
        #         message = response['Messages'][0]
        #         receipt_handle = message['ReceiptHandle']
        #
        #         try:
        #             body = json.loads(message['Body'])
        #             logger.debug(f"Message body: {body}")
        #
        #             # Handle error messages
        #             if 'status' in body and body['status'] == 'error':
        #                 error_message = body.get('error_message', 'An unknown error occurred')
        #                 logger.error(f"YOLO5 job for Detection ID: {detection_id} failed: {error_message}")
        #                 self.delete_message(receipt_handle)
        #                 logger.info(f"Deleted failed detection attempt {message}")
        #                 return chat_id, f"Sorry! >_< Your job has failed: {error_message}. Please try again."
        #
        #             # Handle successful completion messages
        #             detection_id = body.get('JobID')
        #             chat_id = body.get('chat_id')
        #             processed_img_name = body.get('processed_img_name')
        #
        #             if not all([detection_id, chat_id, processed_img_name]):
        #                 raise ValueError(f"Missing required fields in message: {body}")
        #
        #             results = self.fetch_results_from_dynamodb(detection_id)
        #             logger.info(f'Completion message received for JobID: {detection_id}')
        #
        #             processed_image_path = self.download_processed_image_from_s3(
        #                 self.bucket_name,
        #                 processed_img_name,
        #                 local_path=f"/processed_images/{processed_img_name}"
        #             )
        #
        #             self.delete_message(receipt_handle)
        #             logger.info(f"Deleted completion message for JobID: {detection_id}")
        #             return results, processed_image_path
        #
        #         if error_condition:
        #             return chat_id, "Error message"
        #         return chat_id, "Timeout message"
        #
        #         except json.JSONDecodeError:
        #             logger.error(f"Failed to parse message body: {message['Body']}")
        #         except ValueError as ve:
        #             logger.error(str(ve))
        #     except ClientError as e:
        #         logger.error(f"Error while polling SQS: {str(e)}", exc_info=True)
        # logger.warning(f"Job {chat_id} did not complete within the expected time.")
        # return chat_id, f'Sorry, your Detection TimedOut. >_< Please try again.'  # Return None if timeout occurs


    def process_message(self, body, receipt_handle):
        detection_id = body.get('JobID')
        chat_id = body.get('chat_id')
        processed_img_name = body.get('processed_img_name')
        receipt_handle = message['ReceiptHandle']
        logger.debug(f"Message body: {body}")
        if not all([detection_id, chat_id, processed_img_name]):
            raise ValueError(f"Missing required fields in message: {body}")

        try:
            results = self.fetch_results_from_dynamodb(detection_id)
            logger.info(f'Completion message received for JobID: {detection_id}')

            processed_image_path = self.download_processed_image_from_s3(
                self.bucket_name,
                processed_img_name,
                local_path=f"/processed_images/{processed_img_name}"
            )

            self.sqs_listener.delete_message(receipt_handle)
            del self.detection_results[detection_id]
            logger.info(f"Deleted completion message for JobID: {detection_id}")
            return chat_id, (results, processed_image_path)
            # body = json.loads(message['Body'])


            # Handle error messages
            if 'status' in body and body['status'] == 'error':
                error_message = body.get('error_message', 'An unknown error occurred')
                logger.error(f"YOLO5 job for Detection ID: {body.get('JobID')} failed: {error_message}")
                self.delete_message(receipt_handle)
                logger.info(f"Deleted failed detection attempt {message}")
                return body.get('chat_id'), f"Sorry! >_< Your job has failed: {error_message}. Please try again."

            # # Handle successful completion messages
            # detection_id = body.get('JobID')
            # chat_id = body.get('chat_id')
            # processed_img_name = body.get('processed_img_name')

        except json.JSONDecodeError:
            logger.error(f"Failed to parse message body: {message['Body']}")
        except ValueError as ve:
            logger.error(str(ve))
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
        # If we get here, there was an error
            return None, None


    def fetch_results_from_dynamodb(self, detection_id):
        try:
            response = self.table.get_item(
                Key={'JobID': {'S': detection_id}}
            )
            item = response.get('Item', {})
            if not item:
                logger.warning(f"No results found in DynamoDB for JobID: {detection_id}")
            if 'Item' in response:
                detect_results = self.dynamodb_2_dict(item)
                logger.info(f"Successfully retrieved item: {item}")
                return detect_results, 200
            else:
                logger.info(f"No item found with detection_id: {detection_id}")
                return f"No item found with detection_id: {detection_id}", 404
        except Exception as e:
            logger.error(f"Error fetching results from DynamoDB: {e}")
            return {}

    @staticmethod
    def download_processed_image_from_s3(bucket_name, processed_img_name, local_path):
        s3_client = boto3.client('s3')
        # Add 'processed_images/' prefix to the key
        s3_key_with_prefix = f"processed_images/{processed_img_name}"
        # Ensure the directory for local_path exists
        local_directory = os.path.dirname(local_path)
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)
            logger.info(f"Created directory {local_directory} for storing the processed images.")
        try:
            s3_client.download_file(bucket_name, s3_key_with_prefix, local_path)
            logger.info(f"<green>Downloaded processed image {s3_key_with_prefix} from {bucket_name}</green>")
            return local_path
        except ClientError as e:
            logger.error(f"<red>Error downloading processed image from S3: {e}</red>")
            raise

    @staticmethod
    def dynamodb_2_dict(item):
        detected_results = TypeDeserializer()
        return {k: detected_results.deserialize(v) for k, v in item.items()}

    # def delete_message(self, receipt_handle):
    #     try:
    #         self.sqs_client.delete_message(
    #             QueueUrl=self.completion_queue_url,
    #             ReceiptHandle=receipt_handle
    #         )
    #     except ClientError as e:
    #         logger.error(f"Failed to delete message: {str(e)}", exc_info=True)


        def check_for_results(self):
            while True:
                result = self.listen_for_completion.get_processed_message()
                if result:
                    chat_id, message = result
                    # Send result to the user with chat_id
                time.sleep(0.1)  # Small delay to prevent busy-waiting


# Usage
if __name__ == "__main__":
    # Initialize your SQS client, queue URL, and bucket name
    sqs_client = ...
    completion_queue_url = ...
    bucket_name = ...

    # processor = Img(sqs_client, completion_queue_url, bucket_name)
    img_processor = Img()
    img_processor.main_loop()