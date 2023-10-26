import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

CACHE_DIR = "./load_models/models"

# key for user auth
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# key used for send data to the API end points
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
