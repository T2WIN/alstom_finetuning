
# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# --- Directories ---
BASE_DIR = Path(__file__).parent
STORAGE_DIR = Path("/home/grand/alstom_finetuning/vector_store")
DATA_DIR = BASE_DIR / "data/raw"
LOG_FILE = BASE_DIR / "daily_request_log.json"

# --- Models ---
EMBEDDING_MODEL = "../models/qwen3-embed-0.6b"
LLM_FOR_PARSING = "mistralai/mistral-small-3.2-24b-instruct"

# --- API Keys ---
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
JAMBA_API_KEY = os.getenv("JAMBA_API_KEY")

db_path = str(STORAGE_DIR / "qdrant_db_with_filtering")
collection_name = "documents"