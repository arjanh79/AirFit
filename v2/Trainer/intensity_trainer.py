import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from v2.Data.intensity_dataset import IntensityDataset
from v2.Domain.intensity_combinator import IntensityCombinator
from v2.Trainer.intensity_model import IntensityTransformer


class IntensityTrainer:

    def __init__(self, combinator: IntensityCombinator, dataset: IntensityDataset,
                 num_embeddings: dict[str, int], col_names: list[str]):

        self.wc = combinator
        self.ds = dataset
        self.dl = DataLoader(self.ds, batch_size=16, shuffle=True)
        self.num_embeddings = num_embeddings
        self.col_names = col_names

        self.model = IntensityTransformer(num_embeddings=self.num_embeddings, col_names=self.col_names)

        self.optimizer = optim.NAdam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.scheduler = self.get_scheduler()


    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch, (x, y) in enumerate(self.dl):
            self.optimizer.zero_grad()
            intensity = self.model(x)  # (B, 1)

            loss = self.loss_fn(intensity, y.unsqueeze(1))

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            print(f'Batch: {batch}, Batch loss: {loss.item():.5f}')

        loss = total_loss / max(1, len(self.dl))
        return loss


    def fit(self, epochs: int) -> None:
        for epoch in range(1, epochs+1):
            print(f'Epoch {epoch}')
            _  = self.train_one_epoch()  # Output not yet needed
            eval_loss = self.eval()
            print(f'>> Epoch: {epoch}, Epoch loss: {eval_loss:.5f}')
            print('-'*20)

    def eval(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in self.dl:
                intensity = self.model(x)
                loss = self.loss_fn(intensity, y.unsqueeze(1))
                total_loss += loss.item()
        eval_loss = total_loss / len(self.dl)
        self.scheduler.step(eval_loss)

        return eval_loss

    def get_scheduler(self) -> ReduceLROnPlateau:
        scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.75,
            patience=4,
            threshold_mode='rel',
            threshold=1e-3,
            min_lr=1e-6,
            cooldown=3,
        )
        return scheduler
