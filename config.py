import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

CACHE_DIR = "./load_models/models"

# environment
DEVICE_TYPE = os.getenv("DEVICE")

# key for user auth
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# key used for send data to the API end points
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

# IP address that host the Client site or site url:
YOUR_CLIENT_SITE_ADDRESS = os.getenv("YOUR_CLIENT_SITE_ADDRESS")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
