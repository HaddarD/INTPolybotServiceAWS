FROM ultralytics/yolov5:latest-cpu
# Set environment variables
ENV BUCKET_NAME=haddar-aws-project-s3
ENV SQS_JOB_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/019273956931/haddar-AWS-Project-SQS-JobRequests
ENV SQS_COMPLETION_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/019273956931/haddar-AWS-Project-SQS-CompletionNotices
ENV DYNAMO_DB_NAME=haddar-AWS-Project-DynamoDB
# Set the working directory in the container
WORKDIR /usr/src/app
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o yolov5s.pt

COPY . .

CMD ["python3", "app.py"]
