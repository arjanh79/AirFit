
import numpy as np
from datetime import datetime

from matplotlib.style.core import available

from v1.WO import (
    BosuWorkout, ClassicWorkout, ComboWorkout, FocusWorkout, ForgeWorkout,
    HeartWorkout, SingleWorkout, Workout404, CoreWorkout, ShortWorkout, RunningWorkout, ChallengeWorkout,
    OtherDayWorkout
)


def workout_factory(workout_type):
    rnd_gen = np.random.default_rng()

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
        'running': RunningWorkout,
        'challenge': ChallengeWorkout,
        'otherday': OtherDayWorkout
    }

    type_list = list(WORKOUT_MAP.keys()) + ['random'] + ['schedule']

    if workout_type not in type_list:
        print(f'Invalid Workout type: {workout_type}, falling back to \'workout404\'.')
        workout_type = 'workout404'

    print(f'Workout type: {workout_type}')

    if workout_type == 'schedule':
        types = ['challenge', 'otherday', 'heart']
        workout_type = rnd_gen.choice(types, 1)[0]

        print(f'Workout type update: {workout_type}')


    if workout_type == 'random':
        workout_type = rnd_gen.choice(list(WORKOUT_MAP.keys()), 1)[0]
        print(f'Workout type update: {workout_type}')


    WorkoutClass = WORKOUT_MAP.get(workout_type, Workout404)

    return WorkoutClass()
