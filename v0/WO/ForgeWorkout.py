import numpy as np
import pandas as pd
import torch

from sklearn.cluster import KMeans

from dev.v0.AI.biLSTM import AirFitBiLSTM
from dev.v0.WO.BasicWorkout import BasicWorkout

class ForgeWorkout(BasicWorkout):
    def __init__(self):
        super().__init__()
        self.model_location = '../AI/workout_model.pth'
        self.model = AirFitBiLSTM()
        self.model.load_state_dict(torch.load(self.model_location))


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
        result = [self.get_finale()]
        result = result * 3
        result = pd.concat(result, ignore_index=True)
        return result

