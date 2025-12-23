from torch.utils.data import DataLoader

from workout_dataset import WorkoutDataset
from workout_combinator import WorkoutCombinator
from workout_model import WorkoutTransformer

class WorkoutTrainer:
    def __init__(self):
        self.wc = WorkoutCombinator()
        self.ds = WorkoutDataset(self.wc.workouts)
        self.dl = DataLoader(self.ds, batch_size=3, shuffle=True)

        self.model = WorkoutTransformer(len(self.ds.ex_to_id) + 2, d_model=8, n_head=2, num_layers=2, max_len=13)
        for batch, (x, y) in enumerate(self.dl):
            self.model(x)
            break


wt = WorkoutTrainer()