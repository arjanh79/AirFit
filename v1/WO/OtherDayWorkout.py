import pandas as pd

from v1.WO.BasicWorkout import BasicWorkout


class OtherDayWorkout(BasicWorkout):
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

        e1 = df[df['name'] == 'Clean and Press'].sample(n=1)
        e2 = df[df['name'] == 'Step Ups'].sample(n=1)
        e3 = df[df['name'] == 'Push Ups'].sample(n=1)
        e5 = df[df['name'] == 'Bosu Mountain Climbers'].sample(n=1)
        e4 = df[df['name'] == 'Front - Side'].sample(n=1)
        e6 = df[df['name'] == 'Sit Up - Ball'].sample(n=1)



        block = [e1, e2, e3, e4, e5, e6] * 2 + [e1, e2, e3]
        result = pd.concat(block, ignore_index=True)
        return result