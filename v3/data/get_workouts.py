from v3.database.factories import RepositoryFactory
from v3.math.cosine_loss import get_cosine_loss
from v3.schemas.training_data import WorkoutHeader, WorkoutBody, Workout

import torch

class Workouts:
    def __init__(self):
        self.repo = RepositoryFactory.get_repository('sqlite')
        self.training_data = self.repo.get_training_data()
        self.workouts = None


    def generate(self):
        workouts: dict[str, Workout] = {}
        for row in self.training_data[0]:
            workout_id = row[0]
            if workout_id not in workouts:
                loss = get_cosine_loss(len(workouts))
                workouts[workout_id] = Workout(
                    header=WorkoutHeader(row[0], row[1], row[2], row[3], loss),
                    body=[]
                )
            workouts[workout_id].body.append(
                WorkoutBody(row[4], row[5], row[6], row[7], row[8])
            )
        result = list(workouts.values())
        result.sort(key=lambda w: w.header.loss, reverse=True)
        self.workouts = result


    def to_tensor(self):
        x, y = [], []
        for row in self.workouts:
            x_workout = []
            for exercise in row.body:
                x_workout.append([exercise.exercise_id, exercise.exercise_sequence, exercise.core, exercise.reps, exercise.weight_id])
            y_workout = [row.header.workout_intensity, row.header.train_tomorrow, row.header.active_kcal, row.header.loss]

            y.append(y_workout)
            x.append(x_workout)
        return torch.tensor(x), torch.tensor(y)
