import random

from v2.Data.factories import RepositoryFactory
from v2.Trainer.train import WorkoutTrainer


class WorkoutGenerator:
    def __init__(self):
        self.repo = RepositoryFactory.get_repository('sqlite')

        self.workout_token = [1]
        self.workout_exercise = []

