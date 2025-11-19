
import pandas as pd

from v1.WO.BasicWorkout import BasicWorkout


class HeartWorkout(BasicWorkout):
    def __init__(self):
        super().__init__()

    def generate(self):
        warming_up = self.get_warming_up()
        finale = None  # self.get_finale()
        core = self.get_core(warming_up, finale)

        workout, workout_model = self.get_workout(warming_up, core, finale)
        workout, workout_model = self.tune_workout(workout, workout_model)

        print(workout)
        self.estimate_intensity(workout_model, print_output=True)
        self.save_workout(workout)

    def get_core(self, warming_up, finale):

        df = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])

        e1 = df[(df['name'] == 'Step Ups') & (df['weight'] == 12)].sample(n=1)
        e2 = df[(df['name'] == 'Clean and Press') & (df['weight'] == 16)].sample(n=1)
        e3 = df[(df['name'] == 'Bosu Mountain Climbers')].sample(n=1)
        e4 = df[(df['name'] == 'Bent-Over Row') & (df['weight'] == 8)].sample(n=1)
        e5 = df[(df['name'] == 'Bosu Push Up')].sample(n=1)
        e6 = df[(df['name'] == 'Flutter Kicks 4x')].sample(n=1)

        result = [e1, e2, e3, e4, e5, e6] * 2 + [e1, e2, e3]

        result = pd.concat(result, ignore_index=True)
        return result