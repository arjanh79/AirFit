import numpy as np
import pandas as pd
import torch

from dev.v0.AI.dataset import WorkoutDataset
from dev.v0.AI.modelDNN import AirFitDNN
from dev.v0.AI.train import ModelTraining
from dev.v0.DB.factories import RepositoryFactory


class NewWorkout:
    def __init__(self):
        self.repo = RepositoryFactory.get_repository('sqlite')
        self.all_exercises = self.get_all_exercises()
        self.rnd_gen = np.random.default_rng()
        self.model = self.get_model()
        self.mappings = self.get_mappings()

        self.warming_up = self.get_warming_up()
        self.finale = self.get_finale()
        self.core = self.get_core()

        self.workout = self.get_workout()
        print(self.workout)

        self.estimate_intensity()

    def estimate_intensity(self):
        X_embeddings, X_features = self.workout.iloc[:, 0], self.workout.iloc[:, 1:]
        X_embeddings = torch.tensor(X_embeddings.values).long().reshape((1, 20))
        X_features = torch.tensor(X_features.values, dtype=torch.float32).reshape((1, 60))
        y = torch.tensor([[3.5]])
        wl = torch.tensor([[1]])

        dataset = WorkoutDataset(X_embeddings, X_features, y, wl)
        mt = ModelTraining(AirFitDNN(), dataset)
        print(f'\nExpected Intensity: {mt.eval_workout().item():.3f}')



    def get_workout(self):
        workout = pd.concat([self.warming_up, self.core, self.finale], ignore_index=True)
        workout['reps'] = np.where(workout['name'].str.contains('Plank'), 45, 10)

        workout = workout.reindex(range(20), fill_value=0)
        workout.loc[workout['name'] == 0, 'name'] = 'UNK'
        workout['seq_num'] = range(1, 20+1)
        workout['name'] = workout['name'].map(self.mappings).astype(int)

        return workout


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
        exercises = min_exercises.iloc[exercises, :]

        return exercises

    def get_finale(self):
        order = ['Push Ups', 'Squats', 'Clean and Press']
        exercises = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])
        single_weight = exercises.loc[exercises['name'].isin(['Push Ups', 'Clean and Press'])]

        finale = exercises[exercises['name'] == 'Squats'].sample(n=1)
        finale = pd.concat((finale, single_weight))
        finale['name'] = pd.Categorical(finale['name'], categories=order, ordered=True)
        finale = finale.sort_values('name')

        return finale

    def get_core(self):
        core1 = self.get_core_element([self.warming_up])
        core3 = self.get_core_element([self.finale])
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


nwo = NewWorkout()