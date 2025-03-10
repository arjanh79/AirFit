
import pandas as pd
import torch

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
model.load_state_dict(torch.load('../AI/workout_model.pth'))

print(model.embedding.weight.data)

