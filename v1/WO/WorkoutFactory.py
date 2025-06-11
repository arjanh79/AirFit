
import numpy as np

from v1.WO import (
    BosuWorkout, ClassicWorkout, ComboWorkout, FocusWorkout, ForgeWorkout,
    HeartWorkout, SingleWorkout, Workout404, CoreWorkout
)


def workout_factory(workout_type):

    WORKOUT_MAP = {
        'classic': ClassicWorkout,
        'focus': FocusWorkout,
        'forge': ForgeWorkout,
        'combo': ComboWorkout,
        'bosu': BosuWorkout,
        'single': SingleWorkout,
        'workout404': Workout404,
        'heart': HeartWorkout,
        'core': CoreWorkout,
    }

    type_list = list(WORKOUT_MAP.keys()) + ['random']

    if workout_type not in type_list:
        print(f'Invalid Workout type: {workout_type}, falling back to \'workout404\'.')
        workout_type = 'workout404'

    print(f'Workout type: {workout_type}')

    if workout_type == 'random':
        rnd_gen = np.random.default_rng()
        workout_type = rnd_gen.choice(list(WORKOUT_MAP.keys()), 1)[0]
        print(f'Workout type update: {workout_type}')

    WorkoutClass = WORKOUT_MAP.get(workout_type, Workout404)

    return WorkoutClass()
