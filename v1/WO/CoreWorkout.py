
import pandas as pd

from v1.WO.BasicWorkout import BasicWorkout


class CoreWorkout(BasicWorkout):
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
        e1 = df[df['name'] == 'Dead Bug - Static'].sample(n=1)
        e2 = df[df['name'] == 'Plank'].sample(n=1)
        e3 = df[df['name'] == 'Shoulder Taps'].sample(n=1)

        e4 = df[df['name'] == 'Reverse Fly'].sample(n=1)
        e5 = df[df['name'] == 'Bent-Over Row'].sample(n=1)
        e6 = df[df['name'] == 'Squats'].sample(n=1)

        e7 = df[df['name'] == 'Step Ups'].sample(n=1)
        e8 = df[df['name'] == 'Bosu Plank'].sample(n=1)
        e9 = df[df['name'] == 'Push Ups'].sample(n=1)


        result = pd.concat([e1, e2, e3, e4, e5, e6, e7, e8, e9], ignore_index=True)
        return result