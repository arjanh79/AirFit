from torch.utils.data import DataLoader

from v2.Data.intensity_dataset import IntensityDataset
from v2.Domain.intensity_combinator import IntensityCombinator
from v2.Trainer.intensity_model import IntensityTransformer


class IntensityTrainer:

    def __init__(self, combinator: IntensityCombinator, dataset: IntensityDataset, embeddings_dim: dict[str, int]):

        self.wc = combinator
        self.ds = dataset
        self.dl = DataLoader(self.ds, batch_size=16, shuffle=True)
        self.embeddings_dim = embeddings_dim

        self.model = IntensityTransformer(embeddings_dim=self.embeddings_dim)

