import numpy as np
import pandas as pd
import torch

from dev.v0.AI.modelDNN import AirFitDNN
from dev.v0.DB.factories import RepositoryFactory


class NewWorkOut:
    def __init__(self):
        self.repo = RepositoryFactory.get_repository('sqlite')
        self.all_exercises = self.get_all_exercises()
        self.rnd_gen = np.random.default_rng()
        self.model = self.get_model()
        self.get_mappings = self.get_mappings()

    def get_mappings(self):
        mappings = self.repo.get_mapping()[0]
        mappings.insert(0, (0, 'UNK'))
        mappings = {m[1]: c for c, m in enumerate(mappings)}
        return mappings

    def get_all_exercises(self):
        return self.repo.get_all_exercises()

    @staticmethod
    def get_model():
        model = AirFitDNN()
        model.load_state_dict(torch.load('../AI/workout_model.pth'))  # Load the model
        return model

    def get_warming_up(self):
        min_exercises = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])
        min_exercises = min_exercises.groupby('name').min().reset_index()
        min_exercises = min_exercises[~min_exercises['name'].str.contains('Plank|Clean|Push')]

        exercises = self.rnd_gen.choice(len(min_exercises), size=3, replace=False)
        exercises = min_exercises.iloc[exercises, :].reset_index(drop=True)
        exercises['reps'] = 10

        return exercises

    def get_final(self):
        order = ['Push Ups', 'Squats', 'Clean and Press']
        exercises = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])
        single_weight = exercises.loc[exercises['name'].isin(['Push Ups', 'Clean and Press'])]

        finale = pd.DataFrame(self.rnd_gen.choice(exercises[exercises['name'] == 'Squats'], size=1), columns=['name','weight'])
        finale = pd.concat((finale, single_weight), ignore_index=True)
        finale['name'] = pd.Categorical(finale['name'], categories=order, ordered=True)
        finale = finale.sort_values('name').reset_index(drop=True)
        finale['reps'] = 10

        return finale

    def get_core(self):
        exercises = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])
        print(exercises)


nwo = NewWorkOut()
print(nwo.get_warming_up())
print(nwo.get_final())
nwo.get_core()