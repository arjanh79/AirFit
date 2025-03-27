import torch

from sklearn.cluster import KMeans

from dev.v0.AI.biLSTM import AirFitBiLSTM

class TestModel(AirFitBiLSTM):
    def __init__(self):
        super().__init__()


model = TestModel()

model.load_state_dict(torch.load('../workout_model.pth'))
embeddings = model.embedding.weight.data

kmeans = KMeans(n_clusters=3)
x = kmeans.fit_predict(embeddings[1:])

print(x)