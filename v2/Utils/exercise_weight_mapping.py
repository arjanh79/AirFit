import json
from typing import Any
from v2.config import MODEL_PATH
from v2.Data.factories import RepositoryFactory
import pandas as pd

class WeightMappings:
    def __init__(self):
        self.model_dir = MODEL_PATH
        self.repo = RepositoryFactory.get_repository('sqlite')

        self.exercise_ids = {v: int(k) for k, v in self._load_json('exercise_ids.json').items()}


    def _load_json(self, filename: str) -> Any:
        path = self.model_dir / filename
        with open(path, 'r') as f:
            return json.load(f)


    def get_all_weights(self):
        data, cols = self.repo.get_all_weights()
        df = pd.DataFrame(data, columns=cols)
        print(df)

    def print_exercises(self, name):
        print(f'\nExercise "{name}" - ID: {self.exercise_ids[name]}')

wm = WeightMappings()

wm.get_all_weights()
wm.print_exercises('Static Lunge')