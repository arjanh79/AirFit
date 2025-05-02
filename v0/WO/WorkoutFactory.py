
import numpy as np

from dev.v0.WO.ClassicWorkout import ClassicWorkout
from dev.v0.WO.FocusWorkout import FocusWorkout
from dev.v0.WO.ForgeWorkout import ForgeWorkout


def workout_factory(workout_type):

    type_list = ['random', 'classic', 'focus', 'forge']

    if workout_type not in type_list:
        raise ValueError(f'Invalid Workout type: {workout_type}')

    print(f'Workout type: {workout_type}')

    if workout_type == 'random':
        prob = [1] * len(type_list[1:])
        prob[0] += len(type_list[2:])

        rnd_gen = np.random.default_rng()
        workout_type = rnd_gen.choice(type_list[1:], 1, p=[0.5, 0.25, 0.25])
        print(f'Workout type update: {workout_type[0]}')

    if workout_type == 'classic':
        return ClassicWorkout()
    if workout_type == 'focus':
        return FocusWorkout()
    if workout_type == 'forge':
        return ForgeWorkout()


    return None
