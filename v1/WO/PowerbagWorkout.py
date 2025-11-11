import pandas as pd

from v1.WO.BasicWorkout import BasicWorkout

class PowerbagWorkout(BasicWorkout):
    def __init__(self):
        super().__init__()

    def generate(self):
        warming_up = self.get_warming_up()
        finale = None
        core = self.get_core(warming_up, finale)

        workout, workout_model = self.get_workout(warming_up, core, finale)
        workout, workout_model = self.tune_workout(workout, workout_model)

        print(workout)
        self.estimate_intensity(workout_model, print_output=True)
        self.save_workout(workout)

    def get_core(self, warming_up, finale):
        df = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])

        e1 = df[(df['name'] == 'Squats') & (df['weight'] == 25)].sample(n=1)
        e2 = df[(df['name'] == 'Deadlift') & (df['weight'] == 25)].sample(n=1)
        e3 = df[(df['name'] == 'Clean and Press') & (df['weight'] == 25)].sample(n=1)

        f1 = df[(df['name'] == 'Push Ups')]

        result = [e1, e2, e3] * 4 + [f1] * 3

        result = pd.concat(result, ignore_index=True)
        return result