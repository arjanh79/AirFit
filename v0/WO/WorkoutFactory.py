
import numpy as np

from dev.v0.WO.ClassicWorkout import ClassicWorkout
from dev.v0.WO.FocusWorkout import FocusWorkout


def workout_factory(workout_type):

    type_list = ['random', 'classic', 'focus']

    if workout_type not in type_list:
        raise ValueError(f'Invalid Workout type: {workout_type}')

    print(f'Workout type: {workout_type}')

    if workout_type == 'random':
        rnd_gen = np.random.default_rng()
        workout_type = rnd_gen.choice(type_list[1:], 1)

    if workout_type == 'classic':
        return ClassicWorkout()
    if workout_type == 'focus':
        return FocusWorkout()

    return None
