"""Configuration settings for the Workbench application.
"""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SESSIONS_DIR = ARTIFACTS_DIR / "sessions"
EVAL_DIR = BASE_DIR / "evaluation_output"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Security & Processing Constraints
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_QUERY_LIMIT = 100
DEFAULT_QUERY_LIMIT = 10
QUERY_TIMEOUT_SECONDS = 15

# Load .env file if present
env_file = BASE_DIR / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                if k and k not in os.environ:
                    os.environ[k] = v

# Model / LLM Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "hybrid")  # hybrid | rule | gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
