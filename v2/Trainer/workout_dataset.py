import torch
from torch.utils.data import Dataset


class WorkoutDataset(Dataset):
    def __init__(self, workouts: list[list[int]]):
        self.workouts = workouts
        self.start_token = 1
        self.ex_to_id, self.id_to_ex = self.create_tokens()
        self.tokens = self.tokenize()


    def create_tokens(self) -> tuple[dict[int, int], dict[int, int]]:
        exercises = sorted({exercise for workout in self.workouts for exercise in workout})
        ex_to_id = {v: c + 2 for c, v in enumerate(exercises)}
        id_to_ex = {c + 2: v for c, v in enumerate(exercises)}
        return ex_to_id, id_to_ex


    def tokenize(self) -> list[list[int]]:
        return [
            [self.start_token] + [self.ex_to_id[ex] for ex in workout]
            for workout in self.workouts
        ]


    def __len__(self) -> int:
        return len(self.tokens)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.tokens[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

