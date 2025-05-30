import torch

from v1.AI.biLSTM import AirFitBiLSTM
from v1.DB.factories import RepositoryFactory

repo = RepositoryFactory.get_repository('sqlite')
db_mappings = repo.get_mapping()[0]
looks_like = 'Clean'  # Everything new is Clean and Press :)
model_location = '../AI/workout_model.pth'

model = AirFitBiLSTM()
model.load_state_dict(torch.load(model_location))
current_embeddings = model.embedding.weight.data

result = []
for c, v in enumerate(db_mappings):
    if looks_like in v[1]:
        result.append(current_embeddings[c+1])

if len(result) == 0:
    result.append(torch.zeros((3, ), dtype=torch.float32))

result = torch.stack(result)
result = torch.mean(result, dim=0)
result = result.unsqueeze(0)

new_embeddings = torch.concat([current_embeddings, result])

model.embedding = torch.nn.Embedding.from_pretrained(new_embeddings)

torch.save(model.state_dict(), model_location)