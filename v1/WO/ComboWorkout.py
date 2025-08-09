
import pandas as pd
import random
from v1.WO.BasicWorkout import BasicWorkout


class ComboWorkout(BasicWorkout):
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

    def get_finale(self):

        df = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])
        e1 = df[df['name'] == 'Bosu Finale'].sample(n=1)
        block = [e1, e1, e1]

        result = pd.concat(block, ignore_index=True)
        return result


    def get_core(self, warming_up, finale):

        df = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])
        e1 = df[df['name'] == 'Clean and Press'].sample(n=1)
        e2 = df[df['name'] == 'Bent-Over Row'].sample(n=1)
        e3 = df[df['name'] == 'Push Ups'].sample(n=1)
        block = [e1, e2, e3]

        result = []
        for _ in range(4):
            result.extend(block)

        result = pd.concat(result, ignore_index=True)
        return result
