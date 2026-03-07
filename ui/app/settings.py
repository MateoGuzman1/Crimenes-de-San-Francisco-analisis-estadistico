import os
from dotenv import load_dotenv

# Carga .env si existe (en local)
load_dotenv()

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() in {"1", "true", "yes", "y"}
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
