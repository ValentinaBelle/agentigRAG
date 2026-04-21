import os
from dotenv import load_dotenv, find_dotenv

def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_key():
    load_env()
    openai_key = os.getenv('your_key')
    return openai_key
