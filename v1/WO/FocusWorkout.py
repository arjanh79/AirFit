import numpy as np
import pandas as pd
import torch

from sklearn.cluster import KMeans

from dev.v0.AI.biLSTM import AirFitBiLSTM
from dev.v0.WO.BasicWorkout import BasicWorkout

class FocusWorkout(BasicWorkout):
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
        inverse_mapping = {v: k for k, v in self.mappings.items()}
        e_embeddings = self.model.embedding.weight[1:]
        cluster = KMeans(n_clusters=3, n_init=50)
        clusters = cluster.fit_predict(e_embeddings.detach().numpy())

        df = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])
        result = []
        for i in range(3):
            e = self.rnd_gen.choice(np.where(clusters == i)[0], size=1)[0] + 1  # removed UNK
            e_name = inverse_mapping[e]
            result.append(df[df['name'] == e_name].sample(n=1))

        result = result * 3
        result = pd.concat(result, ignore_index=True)
        return result

