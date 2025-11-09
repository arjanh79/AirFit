import datetime

import numpy as np
from v1.WO import (
    HeartWorkout, Workout404, ChallengeWorkout, OtherDayWorkout
)


def workout_factory(workout_type):
    rnd_gen = np.random.default_rng()

    WORKOUT_MAP = {
        'workout404': Workout404,
        'heart': HeartWorkout,
        'challenge': ChallengeWorkout,
        'otherday': OtherDayWorkout
    }

    type_list = list(WORKOUT_MAP.keys()) + ['random'] + ['schedule']

    if workout_type not in type_list:
        print(f'Invalid Workout type: {workout_type}, falling back to \'workout404\'.')
        workout_type = 'workout404'

    print(f'Workout type: {workout_type}')

    if workout_type == 'schedule':
        match datetime.datetime.today().weekday():
            case 0, 2: workout_type = 'random'
            case _: workout_type = 'workout404'

        print(f'Workout type update: {workout_type}')


    if workout_type == 'random':
        workout_type = rnd_gen.choice(list(WORKOUT_MAP.keys()), 1)[0]
        print(f'Workout type update: {workout_type}')


    WorkoutClass = WORKOUT_MAP.get(workout_type, Workout404)

    return WorkoutClass()
