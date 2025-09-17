
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime

from v1.DB.factories import RepositoryFactory
from v1.AI.biLSTM import AirFitBiLSTM

class EmbeddingsPlot:
    def __init__(self):
        self.model_location = '../AI/workout_model.pth'
        self.model = AirFitBiLSTM()
        self.model.load_state_dict(torch.load(self.model_location))
        self.repo = RepositoryFactory.get_repository('sqlite')

    def get_embeddings(self):
        embeddings = self.model.embedding.weight.data
        embeddings = embeddings[1:]
        return embeddings

    def get_pca(self):
        pca = PCA(n_components=2)
        x = pca.fit_transform(self.get_embeddings())
        print(f'Explained Variance:    {pca.explained_variance_ratio_}')
        return x

    def get_tsne(self):
        tsne = TSNE(n_components=2, perplexity=5, random_state=60279)
        x = tsne.fit_transform(self.get_embeddings())
        return x

    def get_db_mappings(self):
        db_mappings = self.repo.get_mapping()[0]
        db_mappings = [(c+1, v[1]) for c, v in enumerate(db_mappings)]
        return db_mappings

    def plot_embeddings(self, data, labels):

        today = self.get_date()

        plt.figure(figsize=(10, 8))
        plt.scatter(data[:, 0], data[:, 1])

        for i, txt in enumerate(labels):
            plt.annotate(str(txt), (data[i, 0], data[i, 1]),
                         textcoords="offset points", xytext=(5, 2), ha='left', fontsize=8)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'Embeddings Plot - {today}')

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

        plt.savefig(f'images/embeddings-{today}.png', dpi=300, bbox_inches='tight')

    def get_date(self):
        return datetime.today().strftime('%Y%m%d')

    def run(self):
        embeddings_2d = self.get_pca()
        labels = [i[1] for i in self.get_db_mappings()]
        self.plot_embeddings(embeddings_2d, labels)




if __name__ == '__main__':
    ep = EmbeddingsPlot()
    ep.run()