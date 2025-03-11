
import numpy as np
import pandas as pd
import torch
import joblib

from dev.v0.AI.dataset import WorkoutDataset
from dev.v0.AI.modelDNN import AirFitDNN
from dev.v0.AI.train import ModelTraining
from dev.v0.DB.factories import RepositoryFactory

repo = RepositoryFactory.get_repository('sqlite')
rnd_gen = np.random.default_rng()

mappings = repo.get_mapping()
df_mapping = pd.DataFrame(mappings[0], columns=mappings[1])
df_mapping = [(v, c+1) for c, v in enumerate(df_mapping['name'])]
df_mapping.insert(0, ('UNK', 0))
df_mapping = dict(df_mapping)

seq_mapping = {i: (i-1)/19  for i in range(1, 20+1)}

exercises = repo.get_all_exercises()
df_exercises = pd.DataFrame(exercises[0], columns=exercises[1])

model = AirFitDNN()
model.load_state_dict(torch.load('../AI/workout_model.pth')) # Load the model


scaler = joblib.load('../AI/scaler.pkl') # Use the same file as used for training

df_exercises_min = df_exercises.groupby('name').min().reset_index()

df_exercises_min = df_exercises_min[~df_exercises_min['name'].str.contains('Plank|Push|Clean')]

warming_up = pd.DataFrame(rnd_gen.choice(df_exercises_min, size=3, replace=False), columns=['name', 'weight'])
warming_up['reps'] = rnd_gen.integers(10, 20+1, size=3)

to_even = warming_up['name'].str.contains('Step Ups|Twist')
is_odd = warming_up['reps'] % 2

warming_up['reps'] = warming_up['reps'] + (to_even & is_odd)
print(warming_up)

warming_up['name'] = warming_up['name'].map(df_mapping)
warming_up.columns = ['e_id', 'weight', 'reps']
warming_up['weight'] = warming_up['weight'].astype(int)

# Dummies... bad decisions were made...
seq_nums = torch.arange(1, 20 + 1).unsqueeze(-1)

warming_up = torch.tensor(warming_up.values, dtype=torch.float32)
warming_up = torch.cat((warming_up, torch.zeros((17, 3))))

warming_up = torch.cat((warming_up, seq_nums), dim=1)

warming_up_e = warming_up[:, 0]
warming_up_f = warming_up[:, 1:]

warming_up_f = warming_up_f.flatten().reshape((-1, 60))
warming_up_f = scaler.transform(warming_up_f)

warming_up_f[:, 2::3] = list(seq_mapping.values())
warming_up_f = torch.tensor(warming_up_f, dtype=torch.float32)

warming_up_e = warming_up_e.reshape((1, -1)).long()

dataset = WorkoutDataset(warming_up_e, warming_up_f, torch.ones((1, 1)), torch.zeros(1))
mt = ModelTraining(AirFitDNN(), dataset)
mt.eval_model()