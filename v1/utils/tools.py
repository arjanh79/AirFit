from datetime import datetime

import torch
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


def get_loss_decay(total_workouts):
    weighted_loss = torch.tensor(np.cumprod([1.075] * (total_workouts - 1)), dtype=torch.float32)
    weighted_loss = torch.cat((torch.ones(1), weighted_loss))
    weighted_loss = (weighted_loss - weighted_loss.min()) / (weighted_loss.max() - weighted_loss.min())
    weighted_loss = torch.clip(weighted_loss, 0.01, 1)
    return weighted_loss
