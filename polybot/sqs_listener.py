import threading
import queue
import json
import time
import logging
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)

class SQSListener:
    def __init__(self, sqs_client, completion_queue_url, bucket_name, message_processor):
        self.sqs_client = sqs_client
        self.completion_queue_url = completion_queue_url
        self.bucket_name = bucket_name
        self.message_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.message_processor = message_processor

    def start_listening(self, num_threads=5):
        for _ in range(num_threads):
            thread = threading.Thread(target=self._worker)
            thread.daemon = True
            thread.start()

    def stop_listening(self):
        self.stop_event.set()

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                response = self.sqs_client.receive_message(
                    QueueUrl=self.completion_queue_url,
                    AttributeNames=['All'],
                    MessageAttributeNames=['All'],
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=10
                )

                if 'Messages' in response and response['Messages']:
                    message = response['Messages'][0]
                    self._process_message(message)
            except ClientError as e:
                logger.error(f"Error while polling SQS: {str(e)}", exc_info=True)

    def _process_message(self, message):
        receipt_handle = message['ReceiptHandle']
        try:
            body = json.loads(message['Body'])
            result = self.message_processor.process_message(body, receipt_handle)
            if result:
                self.message_queue.put(result)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse message body: {message['Body']}")
        except ValueError as ve:
            logger.error(str(ve))

    def delete_message(self, receipt_handle):
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.completion_queue_url,
                ReceiptHandle=receipt_handle
            )
        except ClientError as e:
            logger.error(f"Failed to delete message: {str(e)}", exc_info=True)

    def get_processed_message(self):
        try:
            return self.message_queue.get(block=False)
        except queue.Empty:
            return None

