Here is the config.py for the new project:

# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# --- Directories ---
BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "vector_store"
DATA_DIR = BASE_DIR / "data/raw"
LOG_FILE = BASE_DIR / "daily_request_log.json"

# --- Models ---
EMBEDDING_MODEL = "models/qwen3-embed-0.6b"
LLM_FOR_PARSING = "mistralai/mistral-small-3.2-24b-instruct"

# --- API Keys ---
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
JAMBA_API_KEY = os.getenv("JAMBA_API_KEY")

db_path = str(STORAGE_DIR / "qdrant_db")
collection_name = "documents"

Assuming step 1.2 is already achieved, follow step 1.3.
Ask for human intervention when needed. One example is creating files. Another is asking for existing file that you were not given.
If you ask for information about existing files, wait for the info before doing anything else.
Write a set of tests using pytest for that step. Mock the config but not the database. Implement a creation and cleanup of the test files and database.