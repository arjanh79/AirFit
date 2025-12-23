import random

from v2.Database.factories import RepositoryFactory
from v2.Trainer.train import WorkoutTrainer


class WorkoutGenerator:
    def __init__(self):
        self.repo = RepositoryFactory.get_repository('sqlite')

        self.trainer = WorkoutTrainer()

        self.workout_tokens =