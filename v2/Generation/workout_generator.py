import json
from dataclasses import dataclass
from typing import Any
from collections import Counter, defaultdict
import random

import torch

from v2.Data.factories import RepositoryFactory
from v2.Trainer.workout_model import WorkoutTransformer
from v2.config import MODEL_PATH


@dataclass(frozen=True)
class ModelParams:
    vocab_size: int
    d_model: int
    n_head: int
    num_layers: int
    max_len: int

class WorkoutGenerator:
    def __init__(self, model_dir=MODEL_PATH, weights_file='airfit_model_best.pth'):

        # Create database connection
        self.repo = RepositoryFactory.get_repository('sqlite')

        # File locations
        self.model_dir = model_dir
        self.weights_filename = weights_file

        # Load the mapping at the time of training.
        self.ex_to_id = self._load_int_dict('ex_to_id.json')
        self.id_to_ex = self._load_int_dict('id_to_ex.json')
        self.exercise_ids = {int(k): v for k, v in self._load_json('exercise_ids.json').items()}

        # Load the model.
        params = self._load_model_params('model_params.json')
        self.model = self._build_model(params)
        self._load_weights(self.weights_filename)

        self.model.eval()


    def _load_json(self, filename: str) -> Any:
        path = self.model_dir / filename
        with open(path, 'r') as f:
            return json.load(f)


    def _load_int_dict(self, filename: str) -> dict[int, int]:
        data = self._load_json(filename)
        return {int(k): int(v) for k, v in data.items()}


    def _load_model_params(self, filename: str) -> ModelParams:
        data = self._load_json(filename)
        return ModelParams(vocab_size=int(data['vocab_size']),
                           d_model=int(data['d_model']),
                           n_head=int(data['n_head']),
                           num_layers=int(data['num_layers']),
                           max_len=int(data['max_len']))


    def _build_model(self, p: ModelParams) -> WorkoutTransformer:
        model = WorkoutTransformer(p.vocab_size, p.d_model, p.n_head, p.num_layers, p.max_len)
        return model


    def _load_weights(self, weights_filename: str) -> None:
        weights_path = self.model_dir / weights_filename
        state = torch.load(weights_path, weights_only=True)
        self.model.load_state_dict(state)


    def generate(self):
        exercise_ids = self.select_exercises()
        weights_ids = self.get_weights(exercise_ids)
        workout = self.merge_exercise_weights(exercise_ids, weights_ids)
        print(workout)


    def merge_exercise_weights(self, exercise_ids: list[int], weights_ids: dict[int, int]) -> list[tuple[int, int]]:
        return [(eid, weights_ids[eid]) for eid in exercise_ids]


    def select_exercises(self, length: int=12, temperature: float=1.5) -> list[int]:
        tokens = [1]
        with torch.no_grad():
            for _ in range(length):
                x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
                logits = self.model(x)[0, -1] / temperature

                logits = logits.clone()
                logits[0] = float('-inf') # No padding allowed
                logits[tokens] = float('-inf')  # Including start token

                probs = torch.softmax(logits, dim=-1)
                next_token = int(torch.multinomial(probs, 1).item())
                tokens.append(next_token)
            return self._translate_to_eid(tokens[1:])


    def _translate_to_eid(self, tokens: list[int]) -> list[int]:
        return [self.id_to_ex[token] for token in tokens]


    def get_weights(self, exercise_ids: list[int]) -> dict[int, int]:
        pairs, _ = self.repo.get_exercises(exercise_ids)
        # ['exercise_id', 'weight_id']

        exercise_weight = defaultdict(list)
        for eid, wid in pairs:
            exercise_weight[eid].append(wid)

        chosen_wids = []
        removed_wid = True

        # --- START LOOP HERE

        while removed_wid:
            removed_wid = False
            all_wids = [wid for wids in exercise_weight.values() for wid in wids]
            all_wids = [wid for wid in all_wids if wid not in chosen_wids]

            wid_counter = Counter(all_wids)
            if len(wid_counter) > 0:
                most_common = self.most_common_with_ties(wid_counter)
            else:
                most_common = -1
            chosen_wids.append(most_common)

            exercise_weight_temp = defaultdict(list)
            for eid, wids in exercise_weight.items():
                if most_common in wids:
                    removed_wid = True
                    exercise_weight_temp[eid] = [most_common]
                else:
                    exercise_weight_temp[eid] = wids
            exercise_weight = exercise_weight_temp

        exercise_weight = {k: v[0] for k, v in exercise_weight.items()}
        return exercise_weight


    def most_common_with_ties(self, counter: Counter) -> int:

        # Returns the weight_id which occurs most, using random as a tie-breaker.
        max_count = max(counter.values())
        items = [k for k, v in counter.items() if v == max_count]
        return random.choice(items)



wg = WorkoutGenerator()
wg.generate()