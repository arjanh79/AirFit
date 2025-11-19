import pandas as pd
import random

from v1.WO.BasicWorkout import BasicWorkout


class ChallengeWorkout(BasicWorkout):
    def __init__(self):
        super().__init__()

    def generate(self):
        warming_up = self.get_warming_up()
        finale = None # self.get_finale()
        core = self.get_core(warming_up, finale)

        workout, workout_model = self.get_workout(warming_up, core, finale)
        workout, workout_model = self.tune_workout(workout, workout_model)

        print(workout)
        self.estimate_intensity(workout_model, print_output=True)
        self.save_workout(workout)

    def get_core(self, warming_up, finale):
        df = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])

        a_weight = random.choice([10, 12, 16])
        b_weight = random.choice([5, 8])

        e1 = df[(df['name'] == 'Step Ups') & (df['weight'] == a_weight)].sample(n=1)
        e2 = df[(df['name'] == 'Squats') & (df['weight'] == a_weight)].sample(n=1)
        e4 = df[(df['name'] == 'Bent-Over Row') & (df['weight'] == a_weight)].sample(n=1)
        e5 = df[(df['name'] == 'Bicep Curl') & (df['weight'] == b_weight)].sample(n=1)
        e3 = df[(df['name'] == 'Dumbbell Press') & (df['weight'] == a_weight)].sample(n=1)
        e6 = df[(df['name'] == 'Ab Crunches') & (df['weight'] == b_weight)].sample(n=1)



        block = [e1, e2, e3, e4, e5, e6] * 2 + [e1, e2, e3]
        result = pd.concat(block, ignore_index=True)
        return result