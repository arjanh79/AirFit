from datetime import datetime

import numpy as np

from v1.DB.factories import RepositoryFactory


def get_workout_date():
    repo = RepositoryFactory.get_repository('sqlite')
    workout_timestamp = repo.get_available_workout_dow()

    if workout_timestamp == -1:
        return False

    today_dow = datetime.now().weekday()
    workout_dow = datetime.fromtimestamp(workout_timestamp).weekday()

    return workout_dow == today_dow


def get_weight_decay(len_workout):
    weights = 1.05 ** np.arange(1, len_workout-1)
    weights = 1 + (weights - weights.min()) / (weights.max() - weights.min())
    weights = np.pad(weights, (0, len_workout - len(weights)), mode='edge')
    return weights
