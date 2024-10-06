# Use an official Python runtime as a parent image
FROM python:3.10-slim
# Set environment variables
ENV TELEGRAM_APP_URL=https://haddar.polybot.aws.project.int-devops.click
ENV BUCKET_NAME=haddar-aws-project-s3
ENV SQS_JOB_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/019273956931/haddar-AWS-Project-SQS-JobRequests
ENV SQS_COMPLETION_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/019273956931/haddar-AWS-Project-SQS-CompletionNotices
ENV DYNAMO_DB_NAME=haddar-AWS-Project-DynamoDB
EXPOSE 8443
# Set the working directory in the container
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . /app
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Define the command to run the bot.py
CMD ["python3", "app.py"]

