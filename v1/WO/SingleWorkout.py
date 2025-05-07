import random

import pandas as pd

from dev.v1.WO.BasicWorkout import BasicWorkout


class SingleWorkout(BasicWorkout):
    def __init__(self):
        super().__init__()

    def generate(self):
        warming_up = self.get_warming_up()
        finale = self.get_finale()
        core = self.get_core(warming_up, finale)

        workout, workout_model = self.get_workout(warming_up, core, finale)
        workout, workout_model = self.tune_workout(workout, workout_model)

        print(workout)
        self.estimate_intensity(workout_model, print_output=True)
        self.save_workout(workout)

    def get_core(self, warming_up, finale):
        df = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])
        e = df.sample(n=1)
        result = [e] * 3


        result = pd.concat(result, ignore_index=True)
        return result