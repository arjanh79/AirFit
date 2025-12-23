from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR  / 'v2' / 'Data' / 'airfit.db'
MODEL_PATH = BASE_DIR / 'v2' / 'Modeldata'