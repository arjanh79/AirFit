
import numpy as np

from dev.v1.WO.ClassicWorkout import ClassicWorkout
from dev.v1.WO.FocusWorkout import FocusWorkout
from dev.v1.WO.ForgeWorkout import ForgeWorkout

def workout_factory(workout_type):

    type_list = ['random', 'classic', 'focus', 'forge']

    if workout_type not in type_list:
        raise ValueError(f'Invalid Workout type: {workout_type}')

    print(f'Workout type: {workout_type}')

    if workout_type == 'random':
        rnd_gen = np.random.default_rng()
        workout_type = rnd_gen.choice(type_list[1:], 1, p=[0.4, 0.3, 0.3])
        print(f'Workout type update: {workout_type[0]}')

    if workout_type == 'classic':
        return ClassicWorkout()
    if workout_type == 'focus':
        return FocusWorkout()
    if workout_type == 'forge':
        return ForgeWorkout()

    return None
