from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR  / 'dev' / 'v1' / 'DB' / 'airfit.db'