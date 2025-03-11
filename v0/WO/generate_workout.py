
import pandas as pd
import torch
import joblib

from dev.v0.AI.modelDNN import AirFitDNN
from dev.v0.DB.factories import RepositoryFactory

repo = RepositoryFactory.get_repository('sqlite')

mappings = repo.get_mapping()
df_mapping = pd.DataFrame(mappings[0], columns=mappings[1])
df_mapping = [(c+1, v) for c, v in enumerate(df_mapping['name'])]
df_mapping.insert(0, (0, 'UNK'))

exercises = repo.get_all_exercises()
df_exercises = pd.DataFrame(exercises[0], columns=exercises[1])

model = AirFitDNN()
model.load_state_dict(torch.load('../AI/workout_model.pth')) # Load the model
model.eval() # Put model in eval mode

scaler = joblib.load('../AI/scaler.pkl') # Use the same file as used for training

df_exercises_min = df_exercises.groupby('name').min().reset_index()

print(df_exercises_min)