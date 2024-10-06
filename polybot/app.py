from flask import Flask, request
import os
from bot import Bot
import boto3
import json


app = Flask(__name__)

# TODO load TELEGRAM_TOKEN value from Secret Manager
secrets_client = boto3.client('secretsmanager', region_name='us-east-1')
# Fetch the secret from AWS Secrets Manager
def get_secret(secret_name):
    try:
        response = secrets_client.get_secret_value(SecretId=secret_name)
        secret_json = response['SecretString']  # Get the JSON string
        secret_dict = json.loads(secret_json)  # Parse the JSON string
        return secret_dict['TELEGRAM_BOT_TOKEN']  # Return only the token value
    except Exception as e:
        print(f"Failed to retrieve secret {secret_name}: {e}")
        return None


# Load the secret and set the environment variable
TELEGRAM_TOKEN = get_secret('TELEGRAM_BOT_TOKEN')
# Load the TELEGRAM_APP_URL from environment variables
TELEGRAM_APP_URL = os.environ['TELEGRAM_APP_URL']


@app.route('/', methods=['GET'])
def index():
    return 'Ok', 200


@app.route(f'/{TELEGRAM_TOKEN}/', methods=['POST'])
def webhook():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok', 200

@app.route(f'/loadTest/', methods=['POST'])
def load_test():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok', 200


if __name__ == "__main__":
    bot = Bot(TELEGRAM_TOKEN, TELEGRAM_APP_URL)

    app.run(host='0.0.0.0', port=8443)
