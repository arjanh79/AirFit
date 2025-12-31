import json
from typing import Any
from dataclasses import dataclass
from v2.config import MODEL_PATH


@dataclass
class ExerciseIDMapping:
    name: str
    exercise_id: str
    token_id: str

    def __str__(self):
        return f'{self.name.ljust(25)} ID: {self.exercise_id.rjust(2)} Token: {self.token_id.rjust(2)}'


class Mappings:
    def __init__(self):
        self.model_dir = MODEL_PATH
        self.ex_to_id = self._load_int_dict('ex_to_id.json')
        self.id_to_ex = self._load_int_dict('id_to_ex.json')
        self.exercise_ids = {int(k): v for k, v in self._load_json('exercise_ids.json').items()}


    def _load_json(self, filename: str) -> Any:
        path = self.model_dir / filename
        with open(path, 'r') as f:
            return json.load(f)


    def _load_int_dict(self, filename: str) -> dict[int, int]:
        data = self._load_json(filename)
        return {int(k): int(v) for k, v in data.items()}


    def get_mappings(self) -> None:
        exercises = {name: (exercise_id, self.ex_to_id[exercise_id])  for exercise_id, name in self.exercise_ids.items() if exercise_id in self.ex_to_id.keys()}
        exercises = [ExerciseIDMapping(name, str(ids[0]), str(ids[1])) for name, ids in exercises.items()]
        for exercise in exercises:
            print(exercise)



m = Mappings()
m.get_mappings()
