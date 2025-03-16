import numpy as np
import pandas as pd
import torch

from dev.v0.AI.biLSTM import AirFitBiLSTM
from dev.v0.AI.dataset import WorkoutDataset
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

        self.workout, self.workout_model = self.get_workout()

        self.tune_workout()
        print(self.workout)
        self.estimate_intensity(print_output=True)
        self.save_workout()


    def save_workout(self):
        self.repo.delete_unrated_workouts()
        db_mappings = self.repo.get_mapping()[0]
        db_mappings = {v: k for k, v in db_mappings}
        self.workout['name'] = self.workout['name'].map(db_mappings)
        self.repo.save_workout(self.workout)


    def estimate_intensity(self, print_output=False):
        X_embeddings, X_features = self.workout_model.iloc[:, 0], self.workout_model.iloc[:, 1:]
        X_embeddings = torch.tensor(X_embeddings.values).long().reshape((1, 20))
        X_features = torch.tensor(X_features.values, dtype=torch.float32).reshape((1, 60))
        y = torch.tensor([[3.5]])
        wl = torch.tensor([[1]])

        dataset = WorkoutDataset(X_embeddings, X_features, y, wl)
        mt = ModelTraining(AirFitBiLSTM(), dataset)
        intensity, e_weight = mt.eval_workout()
        intensity = intensity.item()
        if print_output:
            print(f'\nExpected Intensity: {intensity:.3f}')
            print(f'Expected Weight: {e_weight}')
        return intensity, e_weight


    def tune_workout(self):
        intensity, e_weight = self.estimate_intensity(print_output=False)
        e_length = self.workout.shape[0]
        rounds = 0
        while intensity < 3.5 and rounds < 50:
            to_increase = self.workout.sample(n=1, weights=1/(e_weight.squeeze()[:e_length]))
            index = to_increase.index.item()
            to_increase = to_increase.squeeze()
            step_size = 1
            if to_increase['name'] in ['Step Ups', 'Ab Twist']:
                step_size = 2
            if to_increase['name'] in ['Plank', 'Bosu Plank', 'Flutter Kicks 4x']:
                step_size = 5

            self.workout.loc[index, 'reps'] += step_size
            self.workout_model.loc[index, 'reps'] += step_size

            intensity, e_weight = self.estimate_intensity()
            rounds += 1 # Might not be required in the future...
        print(rounds)


    def get_workout(self):
        workout = pd.concat([self.warming_up, self.core, self.finale], ignore_index=True)
        workout['reps'] = np.where(workout['name'].str.contains('Plank'), 45, 10)
        workout['seq_num'] = range(1, 15+1)
        workout_model = workout.reindex(range(20), fill_value=0)
        workout_model.loc[workout_model['name'] == 0, 'name'] = 'UNK'
        workout_model['seq_num'] = range(1, 20+1)
        workout_model['name'] = workout_model['name'].map(self.mappings).astype(int)

        return workout, workout_model


    def get_mappings(self):
        mappings = self.repo.get_mapping()[0]
        mappings.insert(0, (0, 'UNK'))
        mappings = {m[1]: c for c, m in enumerate(mappings)}
        return mappings


    def get_all_exercises(self):
        return self.repo.get_all_exercises()


    @staticmethod
    def get_model():
        model = AirFitBiLSTM()
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
