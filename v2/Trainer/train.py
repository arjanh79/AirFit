
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from workout_dataset import WorkoutDataset
from workout_combinator import WorkoutCombinator
from workout_model import WorkoutTransformer


class WorkoutTrainer:
    def __init__(self):
        self.wc = WorkoutCombinator()
        self.ds = WorkoutDataset(self.wc.workouts)
        self.dl = DataLoader(self.ds, batch_size=16, shuffle=True)

        self.model = WorkoutTransformer(len(self.ds.ex_to_id) + 2, d_model=8, n_head=2, num_layers=2, max_len=13)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
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
            print(f'Batch: {batch}, loss: {loss.item()}')

        loss = total_loss / max(1, len(self.dl))
        return loss

    def predict_next(self, prefix: list[int]) -> int:
        self.model.eval()
        temperature = 0.85

        logits_mask = [0, 1] + prefix

        x = torch.tensor(prefix, dtype=torch.long).unsqueeze(0)
        logits = self.model(x)
        logits = logits[0, -1] / temperature
        logits[logits_mask] = float('-inf')

        probs = torch.softmax(logits, dim=-1)

        next_token = int(torch.multinomial(probs, num_samples=1).item())

        return next_token
