from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR  / 'v1' / 'DB' / 'airfit.db'

TEMPLATES_DIR = BASE_DIR / 'v1' / 'templates'