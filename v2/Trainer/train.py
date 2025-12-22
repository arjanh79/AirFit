from v2.Trainer.workout_dataset import WorkoutDataset
from workout_combinator import WorkoutCombinator

class WorkoutTrainer:
    def __init__(self):
        self.wc = WorkoutCombinator()
        self.ds = WorkoutDataset(self.wc.workouts)




wt = WorkoutTrainer()