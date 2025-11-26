
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import random

from v1.AI.biLSTM import AirFitBiLSTM
from v1.AI.dataset import WorkoutDataset
from v1.AI.train import ModelTraining
from v1.DB.factories import RepositoryFactory

from v1.utils.tools import get_weight_decay


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
        df = pd.DataFrame(self.all_exercises[0], columns=self.all_exercises[1])

        a_weight = random.choice([10, 12])

        e1 = df[(df['name'] == 'Step Ups') & (df['weight'] == a_weight)].sample(n=1)
        e2 = df[(df['name'] == 'Squats') & (df['weight'] == a_weight)].sample(n=1)
        e3 = df[df['name'] == 'Plank'].sample(n=1)
        e4 = df[(df['name'] == 'Bent-Over Row') & (df['weight'] == a_weight)].sample(n=1)
        e5 = df[(df['name'] == 'Dumbbell Press') & (df['weight'] == a_weight)].sample(n=1)

        block = [e1, e2, e3, e4, e5]
        result = pd.concat(block, ignore_index=True)
        return result


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
        workout['reps'] = 8 # default value, leave this even!
        workout.loc[workout['name'].str.contains('Plank', case=False), 'reps'] = 45
        workout.loc[workout['name'].str.contains('Dead Bug', case=False), 'reps'] = 60
        workout.loc[workout['name'].str.contains('Ab', case=False), 'reps'] = 20
        workout.loc[workout['name'].str.contains('Flutter', case=False), 'reps'] = 5

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
        wo_intensity = self.rnd_gen.normal(3.25, 0.1, 1)[0]
        print(f'Target intensity: {wo_intensity:.3f}')

        weights_factor_seq = get_weight_decay(e_length)
        while intensity < wo_intensity and rounds < 50:

            weights_factor_e = 1 / e_weight.squeeze()[:e_length]
            weights_factor_e = 1 + (weights_factor_e - weights_factor_e.min()) / (weights_factor_e.max() - weights_factor_e.min())
            weights_factor = weights_factor_seq + weights_factor_e

            weights_factor = torch.log(weights_factor)

            tau = max(0.1, 1.0 - rounds * 0.01)  # Explore -> Exploit
            weights_factor = torch.softmax(weights_factor / tau, dim=0)

            to_increase = workout.sample(n=1, weights=weights_factor)

            index = to_increase.index.item()
            to_increase = to_increase.squeeze()
            step_size = 2
            if to_increase['name'] in ['Plank', 'Bosu Plank', 'Dead Bug - Static']:
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

