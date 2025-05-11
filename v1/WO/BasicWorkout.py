
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch

from dev.v1.AI.biLSTM import AirFitBiLSTM
from dev.v1.AI.dataset import WorkoutDataset
from dev.v1.AI.train import ModelTraining
from dev.v1.DB.factories import RepositoryFactory



class BasicWorkout(ABC):
    def __init__(self):

        self.model_location = 'AI/workout_model.pth'
        self.model = AirFitBiLSTM()
        self.model.load_state_dict(torch.load(self.model_location))

        self.rnd_gen = np.random.default_rng()

        self.repo = RepositoryFactory.get_repository('sqlite')
        self.all_exercises = self.get_all_exercises()

        self.mappings = self.get_mappings()


    @abstractmethod
    def generate(self):
        pass

    def get_model(self):
        return AirFitBiLSTM().load_state_dict(torch.load(self.model_location))

    def save_workout(self, workout):
        self.repo.delete_unrated_workouts()
        db_mappings = self.repo.get_mapping()[0]
        db_mappings = {v: k for k, v in db_mappings}
        workout['name'] = workout['name'].map(db_mappings)
        self.repo.save_workout(workout)

    def get_all_exercises(self):
        return self.repo.get_all_exercises()

    def get_mappings(self):
        mappings = self.repo.get_mapping()[0]
        mappings.insert(0, (0, 'UNK'))
        mappings = {m[1]: c for c, m in enumerate(mappings)}
        return mappings

    def get_warming_up(self):
        min_exercises = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])
        min_exercises = min_exercises.groupby('name').min().reset_index()
        min_exercises = min_exercises[~min_exercises['name'].str.contains('Plank')]

        exercises = self.rnd_gen.choice(len(min_exercises), size=5, replace=False)
        exercises = min_exercises.iloc[exercises, :]

        return exercises


    def get_finale(self):
        order = ['Squats', 'Push Ups', 'Clean and Press']
        exercises = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])

        pushups = exercises.loc[exercises['name'].isin(['Push Ups'])]
        clean_press = exercises[exercises['name'] == 'Clean and Press'].sample(n=1)
        squats = exercises[exercises['name'] == 'Squats'].sample(n=1)

        finale = pd.concat((squats, pushups, clean_press))
        finale['name'] = pd.Categorical(finale['name'], categories=order, ordered=True)
        finale = finale.sort_values('name')

        return finale

    @abstractmethod
    def get_core(self, warming_up, finale):
        pass

    def get_workout(self, warming_up, core, finale):
        workout = pd.concat([warming_up, core, finale], ignore_index=True)
        # Set  values for reps
        workout['reps'] = 10 # default value, leave this even!
        workout.loc[workout['name'].str.contains('Plank', case=False), 'reps'] = 45
        workout.loc[workout['name'].str.contains('Dead', case=False), 'reps'] = 45
        workout.loc[workout['name'].str.contains('Ab', case=False), 'reps'] = 20

        workout['seq_num'] = range(1, workout.shape[0]+1)
        workout_model = workout.reindex(range(20), fill_value=0)
        workout_model.loc[workout_model['name'] == 0, 'name'] = 'UNK'
        workout_model['seq_num'] = range(1, 20+1)
        workout_model['name'] = workout_model['name'].map(self.mappings).astype(int)

        return workout, workout_model

    def tune_workout(self, workout, workout_model):
        intensity, e_weight = self.estimate_intensity(workout_model, print_output=False)
        e_length = workout.shape[0]
        rounds = 0
        wo_intensity = self.rnd_gen.normal(3.0, 0.25, 1)[0]
        print(f'Target intensity: {wo_intensity:.3f}')
        while intensity < wo_intensity and rounds < 100:
            e_weight = np.where(e_weight < 0.001, 0.001, e_weight)
            # This should direct the focus away from the warming-up
            e_weight[:5] = e_weight[:5] * 10

            weights = 1/(e_weight.squeeze()[:e_length])
            to_increase = workout.sample(n=1, weights=weights)

            index = to_increase.index.item()
            to_increase = to_increase.squeeze()
            step_size = 1
            if to_increase['name'] in ['Step Ups', 'Ab Twist', 'Bosu Mountain Climbers', 'Shoulder Taps', 'Front - Side']:
                step_size = 2
            if to_increase['name'] in ['Plank', 'Bosu Plank', 'Flutter Kicks 4x', 'Dead Bug']:
                step_size = 5

            workout.loc[index, 'reps'] += step_size
            workout_model.loc[index, 'reps'] += step_size

            intensity, e_weight = self.estimate_intensity(workout_model)
            rounds += 1 # Might not be required in the future...(I was wrong)
        print(f'Update rounds: {rounds}')
        return workout, workout_model

    def estimate_intensity(self, workout_model, print_output=False):
        X_embeddings, X_features = workout_model.iloc[:, 0], workout_model.iloc[:, 1:]
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

