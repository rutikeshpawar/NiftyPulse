import os
from pathlib import Path
from dotenv import load_dotenv

# Project root = parent of the src folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Read the .env file and load its values into environment variables
load_dotenv(BASE_DIR / ".env")

# Useful folder paths used across the project
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def get_env(name: str) -> str:
    """Return an environment variable or stop with a clear error."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing '{name}' in .env file")
    return value