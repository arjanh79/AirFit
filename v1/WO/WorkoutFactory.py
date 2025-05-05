
import numpy as np

from dev.v1.WO.BosuWorkout import BosuWorkout
from dev.v1.WO.ClassicWorkout import ClassicWorkout
from dev.v1.WO.ComboWorkout import ComboWorkout
from dev.v1.WO.FocusWorkout import FocusWorkout
from dev.v1.WO.ForgeWorkout import ForgeWorkout
from dev.v1.WO.Workout404 import Workout404


def workout_factory(workout_type):

    type_list = ['random', 'classic', 'focus', 'forge', 'combo', 'bosu', 'workout404']

    if workout_type not in type_list:
        print(f'Invalid Workout type: {workout_type}, falling back to \'workout404\'.')
        workout_type = 'workout404'

    print(f'Workout type: {workout_type}')

    if workout_type == 'random':
        rnd_gen = np.random.default_rng()
        workout_type = rnd_gen.choice(type_list[1:], 1)
        print(f'Workout type update: {workout_type[0]}')

    if workout_type == 'classic':
        return ClassicWorkout()
    if workout_type == 'focus':
        return FocusWorkout()
    if workout_type == 'forge':
        return ForgeWorkout()
    if workout_type == 'combo':
        return ComboWorkout()
    if workout_type == 'bosu':
        return BosuWorkout()
    if workout_type == 'workout404':
        return Workout404()

    return None
