
import pandas as pd
import numpy as np

from dev.v0.WO.BasicWorkout import BasicWorkout


class ClassicWorkout(BasicWorkout):
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
        core1 = self.get_core_element()
        core2 = self.get_core_element([core1])
        core3 = self.get_core_element([core2])
        core = pd.concat([core1, core2, core3], ignore_index=True)
        return core


    def get_core_element(self, exclude=None):
        exercises = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])

        if exclude is not None:
            exclude = pd.concat(exclude)
            available = exercises.loc[~exercises.index.isin(exclude.index)]
        else:
            available = exercises

        diff_ex = 0

        exercise_count = available.groupby(['name']).count().reset_index()
        exercise_count.columns = ['name', 'prob']
        probability = pd.merge(available, exercise_count, on='name')['prob']
        probability = probability / np.sum(probability)

        while diff_ex < 3:
            core = available.sample(n=3, replace=False, weights=probability)
            grouped = core.groupby(['name']).count()
            diff_ex = grouped.shape[0]
        return core

