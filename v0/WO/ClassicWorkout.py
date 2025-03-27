
import pandas as pd

from dev.v0.WO.BasicWorkout import BasicWorkout


class ClassicWorkout(BasicWorkout):

    def __init__(self):
        super().__init__()

    def generate(self):
        warming_up = self.get_warming_up()
        finale = self.get_finale()
        core = self.get_core(warming_up, finale)
        print(warming_up)
        print(finale)
        print(core)

    def get_core(self, warming_up, finale):
        core1 = self.get_core_element([warming_up])
        core3 = self.get_core_element([finale])
        core2 = self.get_core_element([core1, core3])
        core = pd.concat([core1, core2, core3], ignore_index=True)
        return core


    def get_core_element(self, exclude):
        exercises = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])
        exclude = pd.concat(exclude)
        available = exercises.loc[~exercises.index.isin(exclude.index)]
        diff_ex = 0

        while diff_ex < 3:
            core = available.sample(n=3, replace=False)
            grouped = core.groupby(['name']).count()
            diff_ex = grouped.shape[0]
        return core

cw = ClassicWorkout()
cw.generate()