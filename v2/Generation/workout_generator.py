import json
from dataclasses import dataclass
from typing import Any

import torch

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



wg = WorkoutGenerator()
