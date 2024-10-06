import time
from pathlib import Path
from detect import run
import yaml
from loguru import logger
import os
import boto3
from botocore.exceptions import ClientError
import json
from decimal import Decimal
import decimal
from typing import Any, Dict

# AWS Services
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
sqs_client = boto3.client('sqs', region_name='us-east-1')

# Environment variables
bucket_name = os.environ['BUCKET_NAME']
job_queue_url = os.environ['SQS_JOB_QUEUE_URL']
completion_queue_url = os.environ['SQS_COMPLETION_QUEUE_URL']
dynamodb_table_name = dynamodb.Table(os.environ['DYNAMO_DB_NAME'])

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

# img_name = ...  # TODO extract from `message` """(in the process_message function)"""
# chat_id = ...  # TODO extract from `message` """(in the process_message function)"""
# original_img_path = ...  # TODO download img_name from S3, store the local image path in original_img_path
"""(in the download_from_s3 & the process_message functions)"""
# TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).
"""(in the upload_to_s3 & the process_message functions)"""
# TODO store the prediction_summary in a DynamoDB table """(in the store_detection_in_dynamodb & the process_message functions)"""
# TODO perform a GET request to Polybot to `/results` endpoint
"""(Using an SQS notice in the notify_completion & the process_message functions)"""


def convert_decimal_to_float(data: Any) -> Any:
    if isinstance(data, decimal.Decimal):
        return float(data)
    elif isinstance(data, dict):
        return {k: convert_decimal_to_float(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_decimal_to_float(v) for v in data]
    return data


def process_message(response):
    prediction_id = response['Messages'][0]['MessageId']
    body = json.loads(response['Messages'][0]['Body'])
    receipt_handle = response['Messages'][0]['ReceiptHandle']
    img_name = body['image_name']
    chat_id = body['chat_id']

    logger.info(f'Processing: {chat_id} for image: {img_name}')
    try:
        user_img_path = Path(f'images/{img_name}')
        download_from_s3(img_name, str(user_img_path))
        run(
            weights='yolov5s.pt',
            data='data/coco128.yaml',
            source=str(user_img_path),
            project='static/data',
            name=prediction_id,
            save_txt=True
        )
        detection_img_path = Path(f'static/data/{prediction_id}/{img_name}')
        processed_img_name = f'{prediction_id}_{img_name}'
        detection_image_url = upload_to_s3(str(detection_img_path), processed_img_name)
        detect_summary_path = Path(f'static/data/{prediction_id}/labels/{img_name.split(".")[0]}.txt')
        with open(detect_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(l[0])],
                'cx': Decimal(l[1]),
                'cy': Decimal(l[2]),
                'width': Decimal(l[3]),
                'height': Decimal(l[4]),
            } for l in labels]

        # ToDo: Fix structure as per my example
        # Look in file called: yolo_utils.py -> write_to_db
        detection_summary = {
            'prediction_id': str(prediction_id),
            'user_img_path': str(user_img_path),
            'detection_img_path': str(detection_img_path),
            'detection_image_url': detection_image_url,
            'labels': labels
        }
        store_detection_in_dynamodb(detection_summary)

        # ToDo: Before sending to result queue convert to friendly types
        notify_completion(prediction_id, processed_img_name, chat_id)
        sqs_client.delete_message(QueueUrl=job_queue_url, ReceiptHandle=receipt_handle)
        logger.info(f'Processed and deleted message: {receipt_handle}')
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        notify_error(prediction_id, chat_id, str(e))
        sqs_client.delete_message(QueueUrl=job_queue_url, ReceiptHandle=receipt_handle)
        logger.info(f'Deleted failed message: {receipt_handle}')


def store_detection_in_dynamodb(detection_summary):
    try:
        table = dynamodb_table_name
        response = table.put_item(Item=detection_summary)
        logger.info(f"Detection stored in DynamoDB: {json.dumps(detection_summary)}")
        logger.debug(f"DynamoDB response: {response}")
    except ClientError as e:
        logger.error(f"Error writing to DynamoDB: {e}")
        logger.error(f"Error code: {e.response['Error']['Code']}")
        logger.error(f"Error message: {e.response['Error']['Message']}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error writing to DynamoDB: {e}")
        raise


def notify_completion(prediction_id, processed_img_name, chat_id):
    """Send completion notice to Polybot through SQS."""
    try:
        completion_message = json.dumps({
            'JobID': prediction_id,
            'processed_img_name': processed_img_name,
            'chat_id': chat_id
        })
        sqs_client.send_message(
            QueueUrl=completion_queue_url,
            MessageBody=completion_message
        )
        logger.info(f"Sent completion notice for prediction: {prediction_id}")
    except ClientError as e:
        logger.error(f"Error sending completion notice: {e}")
        raise

def consume_queue():
    while True:
        response = sqs_client.receive_message(QueueUrl=job_queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=20)
        if 'Messages' in response:
            process_message(response)


def download_from_s3(s3_key, local_path):
    s3_key_with_prefix = f"user_images/{s3_key}"
    try:
        local_path = Path(local_path)
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
        local_file_path = str(local_path.resolve())
        s3_client.download_file(bucket_name, s3_key_with_prefix, local_file_path)
        logger.info(f'<green>Successfully downloaded {s3_key_with_prefix} from {bucket_name}</green>')
    except ClientError as e:
        logger.error(f'<red>Error downloading from S3: {e}</red>')
        raise

def upload_to_s3(local_path, s3_key):
    s3_key_with_prefix = f"processed_images/{s3_key}"
    try:
        s3_client.upload_file(local_path, bucket_name, s3_key_with_prefix)
        image_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key_with_prefix}"
        logger.info(f'<green>Successfully uploaded {local_path} to s3://{bucket_name}/{s3_key_with_prefix}</green>')
        return image_url
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {e}")
        raise

def notify_error(prediction_id, chat_id, error_message):
    try:
        error_payload = json.dumps({
            'JobID': prediction_id,
            'chat_id': chat_id,
            'status': 'error',
            'error_message': error_message
        })
        sqs_client.send_message(
            QueueUrl=completion_queue_url,
            MessageBody=error_payload
        )
        logger.info(f"Sent error notice for job: {prediction_id}")
    except ClientError as e:
        logger.error(f"Error sending error notice: {e}")
        raise

if __name__ == "__main__":
    consume_queue()
