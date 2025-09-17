import pandas as pd

from v1.WO.BasicWorkout import BasicWorkout
import random


class ShortWorkout(BasicWorkout):
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
        weights = pd.DataFrame(self.repo.get_exercise_count()[0], columns=self.repo.get_exercise_count()[1])
        weights = weights.fillna(0.01)
        weights['WC'] = 1 / weights['WC']
        weights['total'] = weights['WC'] / weights['WC'].sum()
        print(weights)

        exercises = weights['name'].sample(weights = weights['total'], n=3, replace = False)

        result = []

        for e in exercises:
            x = df[df['name'] == e].sample(n=1)
            result.append(x)

        random.shuffle(result)
        result = pd.concat(result, ignore_index=True)
        return result