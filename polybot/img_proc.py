from pathlib import Path
from matplotlib.image import imread, imsave
import random
import os
from loguru import logger
import boto3
from botocore.exceptions import ClientError
import json
import time


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
        self.job_info = {}

        if not self.job_queue_url or not self.completion_queue_url:
            raise EnvironmentError("SQS job or completion queue URL not set in environment variables.")


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

    def upload_and_detect(self, image_path, image_name, chat_id):
        """
        Uploads the image to S3 & sends Yolo5 an SQS job request to detect the image content
        """
        if not image_path:
            raise ValueError("Image path is empty")
        try:
            self.upload_to_s3(image_path, image_name)
            logger.info(f"Successfully uploaded {image_name} to S3")
        except Exception as e:
            logger.exception(f'<red>Error uploading image to S3: {e}</red>')
            raise
        logger.info(f"Starting detection on image: {image_name}")
        # Send job to yolo5 via SQS
        self.send_job_to_sqs(image_name, chat_id)

        # ToDo: Alter this part to work on thread
        # Wait for results using polling mechanism
        results = self.listen_for_completion(chat_id)
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
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {e}")
            raise

    def send_job_to_sqs(self, image_name, chat_id):
        message_body = json.dumps({
            'image_name': image_name,
            'chat_id': chat_id
        })
        self.sqs_client.send_message(
            QueueUrl=self.job_queue_url,
            MessageBody=message_body
        )
        logger.info(f"Sent to YOLO5 for detection: {chat_id}")

    def listen_for_completion(self, chat_id):
        max_attempts = 5  # Adjust as needed
        attempt = 0
        job_id = ""
        # Poll SQS for completion messages
        while attempt < max_attempts:
            logger.info(f"Polling for completion: {chat_id}")
            try:
                response = self.sqs_client.receive_message(
                    QueueUrl=self.completion_queue_url,
                    AttributeNames=['All'],
                    MessageAttributeNames=['All'],
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=10
                )
                logger.debug(f"SQS Response: {response}")
                if 'Messages' not in response or not response['Messages']:
                    logger.info(f"No messages received: {chat_id}, continuing to poll")
                    attempt += 1
                    time.sleep(10)
                    continue

                body = json.loads(response['Messages'][0]['Body'])
                receipt_handle = response['Messages'][0]['ReceiptHandle']
                job_id = body['JobID']
                processed_img_name = body['processed_img_name']
                chat_id = body['chat_id']
                results = self.fetch_results_from_dynamodb()

                # ToDo: fix 'local_path'
                processed_image_path = self.download_processed_image_from_s3(
                    self.bucket_name,
                    processed_image_name,
                    local_path
                )

                return results  # Return the results to bot.py
            except Exception as e:
                logger.error(f"Error while polling SQS: {str(e)}", exc_info=True)

        logger.warning(f"Job {job_id} did not complete within the expected time.")
        return f'Sorry, your Detection TimedOut. >_< Please try again.'  # Return None if timeout occurs

    def fetch_results_from_dynamodb(self, job_id):
        try:
            response = self.table.get_item(
                Key={'JobID': job_id}
            )
            item = response.get('Item', {})
            if not item:
                logger.warning(f"No results found in DynamoDB for job {job_id}")
            return item
        except Exception as e:
            logger.error(f"Error fetching results from DynamoDB: {e}")
            return {}

    @staticmethod
    def download_processed_image_from_s3(self, bucket_name, image_name, local_path):
        s3_client = boto3.client('s3')
        # Add 'processed_images/' prefix to the key
        s3_key_with_prefix = f"processed_images/{image_name}"
        try:
            s3_client.download_file(bucket_name, s3_key_with_prefix, local_path)
            logger.info(f"<green>Downloaded processed image {s3_key_with_prefix} from {bucket_name}</green>")
            return local_path
        except ClientError as e:
            logger.error(f"<red>Error downloading processed image from S3: {e}</red>")
            raise
