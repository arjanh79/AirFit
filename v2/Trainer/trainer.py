import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from v2.Data.workout_dataset import WorkoutDataset
from v2.Domain.workout_combinator import WorkoutCombinator
from v2.Trainer.workout_model import WorkoutTransformer
from v2.config import MODEL_PATH


class WorkoutTrainer:
    def __init__(self, combinator: WorkoutCombinator, dataset: WorkoutDataset):

        self.wc = combinator
        self.ds = dataset
        self.dl = DataLoader(self.ds, batch_size=16, shuffle=True)

        self.model = WorkoutTransformer(len(self.ds.ex_to_id) + 2, d_model=8, n_head=2, num_layers=2, max_len=13)

        self.optimizer = optim.NAdam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()


    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch, (x, y) in enumerate(self.dl):
            self.optimizer.zero_grad()
            logits = self.model(x)  # (B, T, vocab)

            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            print(f'Batch: {batch}, loss: {loss.item():.5f}')

        loss = total_loss / max(1, len(self.dl))
        return loss


    def fit(self, epochs: int=100) -> None:
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}')
            self.train_one_epoch()


    def save_model(self) -> None:
        torch.save(self.model.state_dict(), MODEL_PATH / 'airfit_model.pth')