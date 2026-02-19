import json
from typing import Any
from v2.config import MODEL_PATH


class Mappings:
    def __init__(self):
        self.model_dir = MODEL_PATH
        self.ex_to_id = self._load_int_dict('ex_to_id.json')
        self.id_to_ex = self._load_int_dict('id_to_ex.json')
        self.exercise_ids = {int(k): v for k, v in self._load_json('exercise_ids.json').items()}
        self.exercise_to_token = self.create_mappings()

    def _load_json(self, filename: str) -> Any:
        path = self.model_dir / filename
        with open(path, 'r') as f:
            return json.load(f)


    def _load_int_dict(self, filename: str) -> dict[int, int]:
        data = self._load_json(filename)
        return {int(k): int(v) for k, v in data.items()}

    def create_mappings(self) -> dict[str, int]:
        return {name: self.ex_to_id[exercise_id] for exercise_id, name in self.exercise_ids.items() if exercise_id in self.ex_to_id.keys()}


if __name__ == '__main__':
    m = Mappings()