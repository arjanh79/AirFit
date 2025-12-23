import random

from v2.Data.factories import RepositoryFactory
from v2.Trainer.train import WorkoutTrainer


class WorkoutGenerator:
    def __init__(self):
        self.repo = RepositoryFactory.get_repository('sqlite')

        self.trainer = WorkoutTrainer()
        self.train_model(epochs=500)

        self.workout_token = [1]
        self.workout_exercise = []

        self.get_workout()

        self.eid_mappings = self._get_eid_mappings()
        self.get_exercise_names()

    def get_exercise_names(self):
        exercise_names = [self.eid_mappings[eid] for eid in self.workout_exercise]
        print(exercise_names)



    def _get_eid_mappings(self):
        data, cols = self.repo.get_exercise_ids()
        eid_mappings = {k: v for k, v in data}
        return eid_mappings



    def train_model(self, epochs: int=100) -> None:
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}')
            self.trainer.train_one_epoch()


    def get_workout(self):
        for _ in range(12):
            token, exercise = self.trainer.predict_next(self.workout_token)
            self.workout_token.append(token)
            self.workout_exercise.append(exercise)
        print(f'Workout (tokens): {self.workout_token}')
        print(f'Workout (e_ids) :    {self.workout_exercise}')




wg  = WorkoutGenerator()