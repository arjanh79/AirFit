import torch

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from dev.v0.AI.biLSTM import AirFitBiLSTM

class TestModel(AirFitBiLSTM):
    def __init__(self):
        super().__init__()


model = TestModel()

model.load_state_dict(torch.load('../workout_model.pth'))
embeddings = model.embedding.weight.data


centers = KMeans(n_clusters=3, n_init=50)
emb_centers = centers.fit_predict(embeddings[1:])

pca = PCA(n_components=2)

new_embeddings = pca.fit_transform(embeddings[1:])
new_centers = pca.transform(centers.cluster_centers_)


print(emb_centers)
print(new_embeddings)
print(new_centers)



plt.scatter(new_embeddings[:, 0], new_embeddings[:, 1], s=50, c=emb_centers)
# plt.scatter(new_centers[:, 0], new_centers[:, 1], s=50)
plt.savefig('embeddings.png')