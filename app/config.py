import os
from enum import Enum

from dotenv import load_dotenv


class AuthMode(Enum):
    NO_AUTH = "NO_AUTH"  # No authentication required, server provides its api key to appropriate backends
    CUSTOM_KEY = "CUSTOM_KEY"  # Custom API key required on request, server provides its api key to appropriate backends
    PASS_API_KEY = "PASS_API_KEY"  # Server passes the API key from the request to the backend


class Config(dict):
    def __init__(self):
        super().__init__()

        load_dotenv()

        self.TARGET_API = os.environ.get("TARGET_API", None)
        self.AUTH_MODE = AuthMode(os.environ.get("AUTH_MODE", "PASS_API_KEY"))
        self.AUTH_KEY = os.environ.get("AUTH_KEY", None)
        self.SERVER_PORT = int(os.environ.get("SERVER_PORT", 8000))
        self.LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
        self.MODEL_NAME = os.environ.get("MODEL_NAME", None)
