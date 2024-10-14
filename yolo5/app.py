import time
from pathlib import Path
from detect import run
import yaml
from loguru import logger
import os
import boto3
from botocore.exceptions import ClientError
import json
from typing import Any, Dict
from bson import ObjectId
import uuid
from flask import jsonify
from decimal import Decimal

# AWS Services
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
sqs_client = boto3.client('sqs', region_name='us-east-1')

# Environment variables
bucket_name = os.environ['BUCKET_NAME']
job_queue_url = os.environ['SQS_JOB_QUEUE_URL']
completion_queue_url = os.environ['SQS_COMPLETION_QUEUE_URL']
dynamodb_table_name = dynamodb.Table(os.environ['DYNAMO_DB_NAME'])

detection_results = set()

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


def process_message(response):
    body = json.loads(response['Messages'][0]['Body'])
    receipt_handle = response['Messages'][0]['ReceiptHandle']
    img_name = body['image_name']
    chat_id = body['chat_id']
    detection_id = str(body['JobID'])

    logger.info(f'Processing job for {chat_id} for image: {img_name}')

    user_img_path = Path(f'images/{img_name}')
    try:
        download_from_s3(img_name, str(user_img_path))
        run(
            weights='yolov5s.pt',
            data='data/coco128.yaml',
            source=str(user_img_path),
            project='static/data',
            name=detection_id,
            save_txt=True
        )
        processed_img_name = f'{detection_id}_{img_name}'
        processed_img_path = Path(f'static/data/{detection_id}/{img_name}')
        upload_to_s3(str(processed_img_path), processed_img_name)
        detect_summary_path = Path(f'static/data/{detection_id}/labels/{img_name.split(".")[0]}.txt')
        if not detect_summary_path.exists():
            logger.exception(f'Detection: {detection_id}/{img_name} failed. Detection result not found')
            return f">_< Sorry! Detection: {detection_id}/{img_name} failed. results not found", 404
        if detection_id in detection_results:
            logger.info(f"Skipping doplicate job: {detection_id}")
            return
        detection_results.add(detection_id)
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

        detection_summary = {
            'JobID': str(detection_id),
            'Timestamp': int(time.time()),
            'chat_id': chat_id,
            'processed_img_name': processed_img_name,
            'labels': labels
        }

        db_converted_results = store_detection_in_dynamodb(detection_summary)
        notify_completion(detection_summary, processed_img_name, chat_id)
        sqs_client.delete_message(QueueUrl=job_queue_url, ReceiptHandle=receipt_handle)
        logger.info(f'Processed and deleted message: {receipt_handle}')


    except Exception as e:
        logger.error(f"Error processing message: {e}")
        notify_error(detection_id, chat_id, str(e))
        sqs_client.delete_message(QueueUrl=job_queue_url, ReceiptHandle=receipt_handle)
        logger.info(f'Deleted failed message: {receipt_handle}')


def store_detection_in_dynamodb(detection_summary: Dict[str, Any]):
    table = dynamodb_table_name
    try:
        logger.debug(f"Attempting to store in DynamoDB: {detection_summary}")
        response = table.put_item(Item=detection_summary)
        logger.info(f"Detection stored in DynamoDB: {detection_summary}")
        logger.debug(f"DynamoDB response: {response}")
    except ClientError as e:
        logger.error(f"Error writing to DynamoDB: {e}")
        logger.error(f"Error code: {e.response['Error']['Code']}")
        logger.error(f"Error message: {e.response['Error']['Message']}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error writing to DynamoDB: {e}")
        raise
    return


def notify_completion(detection_summary, processed_img_name, chat_id):
    """Send completion notice to Polybot through SQS."""
    try:
        completion_message = json.dumps({
            'JobID': str(detection_summary['JobID']),
            'Timestamp': str(detection_summary['Timestamp']),
            'processed_img_name': processed_img_name,
            'chat_id': chat_id
        })

        sqs_client.send_message(
            QueueUrl=completion_queue_url,
            MessageBody=completion_message
        )
        logger.info(f"Sent completion notice for detection: {str(detection_summary['JobID'])}")
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
        logger.info(f'<green>Successfully uploaded {local_path} to s3://{bucket_name}/{s3_key_with_prefix}</green>')
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {e}")
        raise


def notify_error(detection_id, chat_id, error_message):
    try:
        error_payload = json.dumps({
            'JobID': detection_id,
            'chat_id': chat_id,
            'status': 'error',
            'error_message': error_message
        })
        sqs_client.send_message(
            QueueUrl=completion_queue_url,
            MessageBody=error_payload
        )
        logger.info(f"Sent error notice to user: {chat_id}")
    except ClientError as e:
        logger.error(f"Error sending error notice: {e}")
        raise


if __name__ == "__main__":
    consume_queue()
