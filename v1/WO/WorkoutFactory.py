
import numpy as np
from datetime import datetime

from v1.WO import (
    BosuWorkout, ClassicWorkout, ComboWorkout, FocusWorkout, ForgeWorkout,
    HeartWorkout, SingleWorkout, Workout404, CoreWorkout, ShortWorkout, RunningWorkout
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
        'short': ShortWorkout,
        'running': RunningWorkout
    }

    type_list = list(WORKOUT_MAP.keys()) + ['random']

    if workout_type not in type_list:
        print(f'Invalid Workout type: {workout_type}, falling back to \'workout404\'.')
        workout_type = 'workout404'

    print(f'Workout type: {workout_type}')

    if workout_type == 'random':
        match datetime.today().weekday():
            case 1 | 3:
                workout_type = 'running'
            case 0 | 2:
                workout_type = 'combo'
            case 4 | 5:
                workout_type = 'short'
            # 6 remains random

    if workout_type == 'random':
        rnd_gen = np.random.default_rng()
        workout_type = rnd_gen.choice(list(WORKOUT_MAP.keys()), 1)[0]
        print(f'Workout type update: {workout_type}')


    WorkoutClass = WORKOUT_MAP.get(workout_type, Workout404)

    return WorkoutClass()
